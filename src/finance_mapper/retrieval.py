"""Hybrid COA candidate retrieval using dense vectors, BM25, and RRF.

This module retrieves candidate Chart of Accounts (COA) accounts for unresolved
source accounts. It does not approve or automatically apply any mapping.
"""

import json
import re
import time
from pathlib import Path

import chromadb
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .mapping import map_trial_balance_accounts
from .schemas import MappingCandidate


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "coa_mapping_candidates"
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lowercase text and return simple word/number tokens for BM25."""
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def collect_unresolved_sources(input_dir: Path) -> list[dict]:
    """Collect unresolved account references and prepare search queries."""

    coa = pd.read_csv(
        input_dir / "chart_of_accounts.csv",
        dtype={"account_code": "string"},
    ).fillna("")

    coa_codes = set(coa["account_code"].str.strip())
    sources: dict[str, dict] = {}

    def add_source(
        code,
        name,
        location: str,
        context: str = "",
    ) -> None:
        code = str(code).strip()
        if not code:
            return

        if code not in sources:
            sources[code] = {
                "source_account_code": code,
                "source_account_name": None,
                "source_locations": [],
                "source_context": [],
            }

        record = sources[code]

        if name and str(name).strip():
            new_name = str(name).strip()
            if record["source_account_name"] is None:
                record["source_account_name"] = new_name
            elif record["source_account_name"] != new_name:
                additional_name = f"Additional source name: {new_name}"
                if additional_name not in record["source_context"]:
                    record["source_context"].append(additional_name)

        if location not in record["source_locations"]:
            record["source_locations"].append(location)

        if context and context not in record["source_context"]:
            record["source_context"].append(context)

    # Current trial-balance accounts without an accepted target.
    decisions = map_trial_balance_accounts(input_dir)
    for decision in decisions:
        if decision.target_account_code is None:
            add_source(
                decision.source_account_code,
                decision.source_account_name,
                "trial_balance.csv",
                decision.reason,
            )

    # Historical accounts absent from the current COA.
    prior = pd.read_csv(
        input_dir / "prior_period_tb.csv",
        dtype={"account_code": "string"},
    ).fillna("")

    for row in prior.itertuples(index=False):
        code = str(row.account_code).strip()
        if code and code not in coa_codes:
            add_source(
                code,
                row.account_name,
                "prior_period_tb.csv",
                "Historical account absent from the current COA.",
            )

    # Account references in manual adjustments that lack a COA entry.
    adjustments_path = input_dir / "manual_adjustments.json"
    with adjustments_path.open(encoding="utf-8") as file:
        adjustments = json.load(file)

    for entry in adjustments.get("entries", []):
        for line in entry.get("lines", []):
            code = str(line.get("account", "")).strip()
            if code and code not in coa_codes:
                add_source(
                    code,
                    None,
                    f"manual_adjustments.json:{entry.get('id', 'unknown')}",
                    (
                        f"{entry.get('description', '')}; "
                        f"memo: {line.get('memo', '')}"
                    ),
                )

    # Keep descriptive context for semantic search, but omit generic reasons
    # that explain why the record was flagged rather than what it represents.
    generic_context = (
        "No exact account-code match exists",
        "Historical account absent from the current COA.",
    )

    for record in sources.values():
        semantic_context = [
            context
            for context in record["source_context"]
            if not context.startswith(generic_context)
        ]
        parts = [
            record["source_account_name"] or "",
            *semantic_context,
        ]
        record["query_text"] = " ".join(
            part.strip()
            for part in parts
            if part and part.strip()
        )

    return list(sources.values())


def _build_coa_document(row, parent_names: dict[str, str]) -> str:
    """Build one searchable document for a single COA account."""

    code = str(row["account_code"]).strip()
    parent_code = str(row["parent_code"]).strip()

    fields = [
        f"Account code: {code}",
        f"Account name: {row['account_name']}",
        f"Account type: {row['account_type']}",
        f"Statement: {row['statement']}",
        f"Cash-flow category: {row['cf_category']}",
        f"Normal balance: {row['normal_balance']}",
    ]

    if parent_code:
        fields.append(
            f"Parent account: {parent_names.get(parent_code, parent_code)}"
        )

    return ". ".join(fields)


def retrieve_coa_candidates(
    input_dir: Path,
    output_dir: Path,
    top_k: int = 3,
) -> dict:
    """Retrieve COA candidates with dense search + BM25 fused by RRF."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    total_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = collect_unresolved_sources(input_dir)

    coa = pd.read_csv(
        input_dir / "chart_of_accounts.csv",
        dtype={
            "account_code": "string",
            "parent_code": "string",
        },
    ).fillna("")

    coa["account_code"] = coa["account_code"].str.strip()
    coa["parent_code"] = coa["parent_code"].str.strip()

    # Do not offer header accounts or accounts with children as targets.
    parent_codes = set(coa.loc[coa["parent_code"].ne(""), "parent_code"])
    candidates = coa.loc[
        coa["account_type"].str.casefold().ne("header")
        & ~coa["account_code"].isin(parent_codes)
        & coa["account_code"].ne("")
    ].copy()

    if candidates.empty:
        raise ValueError("No eligible COA posting-account candidates found.")

    parent_names = dict(zip(coa["account_code"], coa["account_name"]))

    # One COA account = one searchable document; no arbitrary text chunking.
    candidate_documents = [
        _build_coa_document(row, parent_names)
        for _, row in candidates.iterrows()
    ]
    candidate_ids = [
        f"{row['account_code']}__{i}"
        for i, (_, row) in enumerate(candidates.iterrows())
    ]
    metadata = [
        {
            "account_code": str(row["account_code"]),
            "account_name": str(row["account_name"]),
            "account_type": str(row["account_type"]),
            "statement": str(row["statement"]),
            "cf_category": str(row["cf_category"]),
            "normal_balance": str(row["normal_balance"]),
        }
        for _, row in candidates.iterrows()
    ]

    # Dense embeddings for semantic retrieval.
    model_start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    model_load_seconds = time.perf_counter() - model_start

    embed_start = time.perf_counter()
    candidate_embeddings = model.encode(
        candidate_documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_texts = [source["query_text"] for source in sources]
    query_embeddings = (
        model.encode(
            query_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if query_texts
        else []
    )
    embedding_seconds = time.perf_counter() - embed_start

    # Build a BM25 keyword index over the same eligible COA documents.
    bm25_corpus = [_tokenize(document) for document in candidate_documents]
    bm25 = BM25Okapi(bm25_corpus)

    # Rebuild the local Chroma collection so this run uses the current COA.
    index_dir = output_dir / "chroma_coa_index"
    client = chromadb.PersistentClient(path=str(index_dir))
    existing_names = {
        collection if isinstance(collection, str) else collection.name
        for collection in client.list_collections()
    }
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=candidate_ids,
        documents=candidate_documents,
        embeddings=candidate_embeddings.tolist(),
        metadatas=metadata,
    )

    candidate_results = []
    dense_search_seconds = 0.0
    bm25_fusion_seconds = 0.0
    retrieval_start = time.perf_counter()

    if sources:
        # Fetch the full dense ranking so RRF can combine ranks consistently.
        dense_start = time.perf_counter()
        dense_results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=len(candidate_ids),
            include=["metadatas", "distances"],
        )
        dense_search_seconds = time.perf_counter() - dense_start

        fusion_start = time.perf_counter()
        for source_index, source in enumerate(sources):
            dense_ids = dense_results["ids"][source_index]
            dense_distances = dense_results["distances"][source_index]
            dense_metas = dense_results["metadatas"][source_index]

            dense_rank_by_id = {
                candidate_id: rank
                for rank, candidate_id in enumerate(dense_ids, start=1)
            }
            distance_by_id = dict(zip(dense_ids, dense_distances))
            metadata_by_id = dict(zip(dense_ids, dense_metas))

            # BM25 score is used only to form a keyword ranking. It is not
            # combined arithmetically with cosine similarity.
            query_tokens = _tokenize(source["query_text"])
            bm25_scores = (
                bm25.get_scores(query_tokens)
                if query_tokens
                else [0.0] * len(candidate_ids)
            )
            bm25_ranked_indexes = [
                (index, float(score))
                for index, score in enumerate(bm25_scores)
                if score > 0
            ]
            bm25_ranked_indexes.sort(key=lambda item: (-item[1], item[0]))
            bm25_rank_by_id = {
                candidate_ids[index]: rank
                for rank, (index, _) in enumerate(
                    bm25_ranked_indexes,
                    start=1,
                )
            }

            # Reciprocal Rank Fusion combines the two ranked lists.
            fused_scores = {}
            for candidate_id in candidate_ids:
                dense_rank = dense_rank_by_id[candidate_id]
                score = 1.0 / (RRF_K + dense_rank)
                bm25_rank = bm25_rank_by_id.get(candidate_id)
                if bm25_rank is not None:
                    score += 1.0 / (RRF_K + bm25_rank)
                fused_scores[candidate_id] = score

            ranked_ids = sorted(
                candidate_ids,
                key=lambda candidate_id: (
                    -fused_scores[candidate_id],
                    dense_rank_by_id[candidate_id],
                ),
            )[: min(top_k, len(candidate_ids))]

            for rank, candidate_id in enumerate(ranked_ids, start=1):
                meta = metadata_by_id[candidate_id]
                distance = float(distance_by_id[candidate_id])
                dense_rank = dense_rank_by_id[candidate_id]
                bm25_rank = bm25_rank_by_id.get(candidate_id)
                lexical_rank_text = (
                    str(bm25_rank)
                    if bm25_rank is not None
                    else "no positive lexical match"
                )

                # Chroma cosine distance = 1 - cosine similarity. This field
                # remains cosine similarity; the displayed rank uses RRF.
                similarity = max(-1.0, min(1.0, 1.0 - distance))

                candidate_results.append(
                    MappingCandidate(
                        source_account_code=source["source_account_code"],
                        source_account_name=source["source_account_name"],
                        source_locations=source["source_locations"],
                        source_context=source["source_context"],
                        candidate_account_code=meta["account_code"],
                        candidate_account_name=meta["account_name"],
                        candidate_account_type=meta["account_type"],
                        similarity_score=similarity,
                        rank=rank,
                        evidence=[
                            f"COA record ID: {candidate_id}",
                            f"Account name: {meta['account_name']}",
                            f"Account type: {meta['account_type']}",
                            f"Statement: {meta['statement']}",
                            f"Cash-flow category: {meta['cf_category']}",
                            f"Normal balance: {meta['normal_balance']}",
                            f"Dense rank: {dense_rank}",
                            f"BM25 rank: {lexical_rank_text}",
                            f"RRF score: {fused_scores[candidate_id]:.6f}",
                        ],
                    )
                )

        bm25_fusion_seconds = time.perf_counter() - fusion_start

    retrieval_and_fusion_seconds = time.perf_counter() - retrieval_start
    total_seconds = time.perf_counter() - total_start

    report = {
        "retrieval_method": "Dense vectors + BM25 + Reciprocal Rank Fusion",
        "embedding_model": MODEL_NAME,
        "lexical_retriever": "BM25Okapi",
        "fusion_method": f"RRF with k={RRF_K}",
        "candidate_accounts_indexed": len(candidate_ids),
        "unresolved_source_accounts": len(sources),
        "candidate_count": len(candidate_results),
        "latency_seconds": {
            "model_load": round(model_load_seconds, 4),
            "embedding_generation": round(embedding_seconds, 4),
            "vector_search": round(dense_search_seconds, 4),
            "bm25_and_fusion": round(bm25_fusion_seconds, 4),
            "retrieval_and_fusion": round(retrieval_and_fusion_seconds, 4),
            "total": round(total_seconds, 4),
        },
        "llm_calls": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_api_cost_usd": 0.0,
        "note": (
            "Retrieval-only stage: no LLM calls were made. The rank is based "
            "on Reciprocal Rank Fusion. similarity_score is cosine similarity, "
            "not a calibrated mapping confidence or accounting approval. "
            "All candidates require human review."
        ),
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in candidate_results
        ],
    }

    report_path = output_dir / "mapping_candidates.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
