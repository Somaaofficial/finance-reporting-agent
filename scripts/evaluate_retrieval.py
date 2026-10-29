import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = ROOT / "output" / "mapping_candidates.json"
GOLD_PATH = ROOT / "data" / "evaluation" / "gold_candidates.json"
REPORT_PATH = ROOT / "output" / "retrieval_evaluation.json"

TOP_K = 3


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    retrieval_report = load_json(RESULTS_PATH)
    gold_report = load_json(GOLD_PATH)

    # Group candidate results by the source account.
    candidates_by_source = {}

    for candidate in retrieval_report["candidates"]:
        code = str(candidate["source_account_code"])
        candidates_by_source.setdefault(code, []).append(candidate)

    for candidates in candidates_by_source.values():
        candidates.sort(key=lambda item: item["rank"])

    evaluated_queries = []
    excluded_queries = []

    # Compare the retrieved results against our reference judgments.
    for case in gold_report["cases"]:
        source_code = str(case["source_account_code"])
        candidates = candidates_by_source.get(source_code, [])
        retrieved_codes = [
            str(item["candidate_account_code"])
            for item in candidates[:TOP_K]
        ]

        if not case["ranking_evaluable"]:
            excluded_queries.append(source_code)
            continue

        relevant_codes = set(
            str(code)
            for code in case["relevant_candidate_codes"]
        )

        if not relevant_codes:
            raise ValueError(
                f"{source_code} is marked evaluable but has no relevant codes."
            )

        hits = [
            code for code in retrieved_codes
            if code in relevant_codes
        ]

        precision_at_k = len(hits) / TOP_K
        recall_at_k = len(hits) / len(relevant_codes)

        first_relevant_rank = next(
            (
                rank
                for rank, code in enumerate(retrieved_codes, start=1)
                if code in relevant_codes
            ),
            None,
        )

        reciprocal_rank = (
            1 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        evaluated_queries.append({
            "source_account_code": source_code,
            "retrieved_top_k": retrieved_codes,
            "relevant_candidate_codes": sorted(relevant_codes),
            "hits": hits,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "reciprocal_rank": reciprocal_rank,
        })

    if not evaluated_queries:
        raise ValueError("No queries have usable relevance labels.")

    query_count = len(evaluated_queries)

    # Check that retrieval suggestions remain non-authoritative.
    all_candidates = retrieval_report["candidates"]
    safe_candidates = [
        item for item in all_candidates
        if item.get("status") == "candidate_only"
        and item.get("requires_human_review") is True
    ]

    metrics = {
        "top_k": TOP_K,
        "ranking_evaluable_queries": query_count,
        "excluded_queries_without_verified_target": excluded_queries,
        "macro_precision_at_k": sum(
            item["precision_at_k"] for item in evaluated_queries
        ) / query_count,
        "macro_recall_at_k": sum(
            item["recall_at_k"] for item in evaluated_queries
        ) / query_count,
        "mean_reciprocal_rank_at_k": sum(
            item["reciprocal_rank"] for item in evaluated_queries
        ) / query_count,
        "candidate_safety": {
            "total_candidates": len(all_candidates),
            "requiring_human_review": len(safe_candidates),
            "safety_rate": (
                len(safe_candidates) / len(all_candidates)
                if all_candidates else None
            ),
            "unsafe_candidates": [
                item for item in all_candidates
                if not (
                    item.get("status") == "candidate_only"
                    and item.get("requires_human_review") is True
                )
            ],
        },
        "per_query_results": evaluated_queries,
        "evaluation_limitations": gold_report.get("limitations", ""),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Retrieval evaluation completed.")
    print(f"Evaluated queries: {query_count}")
    print(
        f"Precision@{TOP_K}: "
        f"{metrics['macro_precision_at_k']:.4f}"
    )
    print(
        f"Recall@{TOP_K}: "
        f"{metrics['macro_recall_at_k']:.4f}"
    )
    print(
        f"MRR@{TOP_K}: "
        f"{metrics['mean_reciprocal_rank_at_k']:.4f}"
    )
    print(
        "Candidate safety rate: "
        f"{metrics['candidate_safety']['safety_rate']:.2%}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()