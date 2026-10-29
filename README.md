# Finance Reporting Agent — TB-to-COA Mapping Prototype

## 1. Project Overview

This project is a prototype for the AI Agentic Engineer take-home assignment. It implements one selected slice of a larger financial reporting platform: **Trial Balance to Chart of Accounts (COA) mapping**.

The prototype uses deterministic account matching, data-quality profiling, dense vector retrieval, BM25 keyword retrieval, Reciprocal Rank Fusion (RRF), and retrieval evaluation.

It is designed to handle messy financial input data without silently correcting accounting issues or automatically approving uncertain mappings.

## 2. Implemented Features

- **Data-quality profiling:** Identifies issues across the supplied trial balance, chart of accounts, prior-period trial balance, FX rates, and manual adjustments.
- **Deterministic mapping:** Matches account codes against eligible COA records and flags ambiguous, unmapped, or review-required accounts.
- **Dense retrieval:** Uses Sentence Transformers embeddings and ChromaDB with cosine distance to retrieve candidate accounts.
- **Keyword retrieval:** Uses BM25 to rank candidates based on matching terms.
- **Hybrid ranking:** Combines dense and lexical rankings using Reciprocal Rank Fusion.
- **Retrieval evaluation:** Calculates Precision@3, Recall@3, and MRR@3 against a small provisional evaluation dataset.
- **Machine-readable outputs:** Writes profiling reports, mapping decisions, candidate mappings, and evaluation results to JSON and CSV files.

## 3. Technology Stack

- Python
- Pandas
- Pydantic
- Sentence Transformers
- ChromaDB
- BM25 (`rank-bm25`)
- pytest

## 4. Project Structure

```text
finance-reporting-agent/
├── inputs/                 # Supplied financial input files
├── data/                   # Evaluation data
├── output/                 # Generated reports and local index
├── scripts/
│   ├── profile_inputs.py
│   ├── run_exact_mapper.py
│   ├── run_candidate_retrieval.py
│   ├── evaluate_retrieval.py
│   └── check_schema.py
├── src/
│   └── finance_mapper/
│       ├── mapping.py
│       ├── profiling.py
│       ├── retrieval.py
│       └── schemas.py
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. Setup

Run the following commands in Windows PowerShell from the project root.

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Make the source package available to Python:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

The `rank-bm25` dependency must be listed in `requirements.txt` because the hybrid retrieval implementation uses it.

## 6. Run the Prototype

Run the scripts in this order.

### Step 1: Profile the input data

```powershell
python scripts\profile_inputs.py
```

Generates:

`output/data_quality_report.json`

### Step 2: Run deterministic account mapping

```powershell
python scripts\run_exact_mapper.py
```

Generates:

- `output/mapping_decisions.json`
- `output/mapping_decisions.csv`

### Step 3: Retrieve COA candidates using hybrid retrieval

```powershell
python scripts\run_candidate_retrieval.py
```

Generates:

`output/mapping_candidates.json`

The first execution may download the embedding model from Hugging Face.

### Step 4: Evaluate retrieval quality

```powershell
python scripts\evaluate_retrieval.py
```

Generates:

`output/retrieval_evaluation.json`

### Run tests

```powershell
python -m pytest -q
```

## 7. Retrieval Evaluation

The evaluation script reports:

- **Precision@3:** The fraction of the top-three returned candidates that are labelled relevant.
- **Recall@3:** The fraction of labelled relevant candidates retrieved in the top three.
- **MRR@3:** The reciprocal rank of the first relevant candidate, averaged across evaluable queries.

The current provisional benchmark has two ranking-evaluable queries. The account `9999` is excluded from ranking metrics because no verified target mapping is available.

Current observed metrics:

| Metric | Result |
|---|---:|
| Precision@3 | 0.3333 |
| Recall@3 | 1.0000 |
| MRR@3 | 1.0000 |
| Candidate safety rate | 100% |

These results are a prototype smoke test, not evidence of production mapping accuracy. The reference labels require independent finance review, and the benchmark must be expanded before drawing broader conclusions.

## 8. Safety and Accounting Controls

- Preserve the original supplied input files.
- Do not automatically map an unknown account based only on its similarity score.
- Treat retrieved accounts as candidates, not approved mappings.
- Route ambiguous and unsupported mappings for human review.
- Do not silently repair unbalanced journal entries.
- Do not substitute an average FX rate when a required period-end rate is missing.
- Preserve duplicate source rows until their accounting meaning is confirmed.
- Do not treat balanced totals alone as proof of accounting correctness.

## 9. Scope and Limitations

This is a local prototype of the TB-to-COA mapping slice, not a complete financial reporting platform.

The current implementation includes data profiling, deterministic mapping, hybrid retrieval, and retrieval evaluation. It does not yet implement an LLM recommendation node, RAGAS evaluation, a complete approval interface, or generation of the four financial statements.

Retrieval scores are ranking signals and are not calibrated probabilities. Human-review flags in the output do not themselves constitute a completed approval workflow.

## 10. Future Improvements

- Expand the finance-reviewed evaluation dataset.
- Measure dense-only and hybrid retrieval on the same independently reviewed benchmark.
- Add structured LLM-assisted recommendations with candidate validation and abstention.
- Evaluate generated recommendations for groundedness and relevance.
- Implement a persistent, auditable human-review workflow.
- Extend the architecture to adjustments, FX translation, consolidation, statement generation, and reconciliation.

## 11. Use of AI Tools

AI assistance was used for design exploration, code scaffolding, debugging, test development, and documentation. The implementation was run locally, and its observed behavior and test results were used to verify the prototype. Accounting decisions remain subject to deterministic validation and finance review.
