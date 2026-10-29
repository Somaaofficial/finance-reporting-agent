# Finance Reporting Agent — TB-to-COA Mapping Prototype

## 1. Project Overview

This project implements one selected slice of the AI Agentic Engineer take-home assignment: **Trial Balance (TB) to Chart of Accounts (COA) mapping**.

The prototype combines deterministic account matching, data-quality profiling, hybrid candidate retrieval, LLM-assisted recommendations, and a human-in-the-loop review workflow.

The system uses Python, Sentence Transformers, ChromaDB, BM25, Reciprocal Rank Fusion (RRF), Gemini, LangGraph, Pydantic, and FastAPI.

The design prioritizes traceability and accounting safety. Retrieved accounts are treated as candidates rather than automatically approved mappings. Uncertain or unsupported cases can be escalated for human review.

## 2. Implemented Features

- **Data-quality profiling:** Identifies potential issues in the supplied trial balance, chart of accounts, prior-period trial balance, FX rates, and manual adjustments.
- **Deterministic mapping:** Matches account codes against eligible COA records and flags ambiguous, unmapped, or review-required accounts.
- **Dense retrieval:** Uses Sentence Transformers embeddings and ChromaDB with cosine distance to retrieve candidate accounts.
- **Keyword retrieval:** Uses BM25 to rank candidates based on matching terms.
- **Hybrid retrieval:** Combines dense and lexical rankings using Reciprocal Rank Fusion.
- **Retrieval evaluation:** Calculates Precision@3, Recall@3, and MRR@3 using a small provisional evaluation dataset.
- **LLM-assisted recommendations:** Uses Gemini to recommend mappings based on retrieved candidates and structured output schemas.
- **Candidate validation:** Checks that a recommended target belongs to the retrieved candidate set rather than accepting an arbitrary COA code.
- **LangGraph workflow:** Organizes candidate retrieval, recommendation, human review, and finalization into graph nodes.
- **Human-in-the-loop review:** Supports approve, reject, and quarantine decisions for unresolved mapping cases.
- **FastAPI endpoints:** Exposes API endpoints to run the workflow, submit review decisions, check health, and retrieve the latest report.
- **Observability:** Supports LangSmith tracing for inspecting workflow execution and model calls.
- **Machine-readable outputs:** Writes profiling reports, mapping decisions, candidate mappings, evaluation results, and agent-review results to JSON and CSV files.

## 3. Technology Stack

- Python
- Pandas
- Pydantic
- Sentence Transformers
- ChromaDB
- BM25 (`rank-bm25`)
- LangChain
- LangGraph
- Gemini through `langchain-google-genai`
- FastAPI and Uvicorn
- LangSmith
- pytest

## 4. Project Structure

```text
finance-reporting-agent/
├── inputs/                   # Supplied financial input files
├── data/
│   └── evaluation/           # Provisional retrieval evaluation labels
├── docs/
│   ├── architecture.md
│   ├── reflection.md
│   └── submission_checklist.md
├── output/                   # Generated reports and evaluation results
├── scripts/
│   ├── profile_inputs.py
│   ├── run_exact_mapper.py
│   ├── run_candidate_retrieval.py
│   ├── evaluate_retrieval.py
│   ├── run_agent_review.py
│   └── check_schema.py
├── src/
│   └── finance_mapper/
│       ├── agent_graph.py    # LangGraph workflow and human review
│       ├── api.py            # FastAPI application
│       ├── mapping.py        # Deterministic mapping
│       ├── profiling.py      # Input data-quality profiling
│       ├── retrieval.py      # Dense, BM25, and hybrid retrieval
│       └── schemas.py        # Pydantic schemas
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

The local ChromaDB index is rebuildable and is excluded from version control.

## 5. Setup

Run these commands in Windows PowerShell from the project root.

### Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Configure the Python source path

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### Configure environment variables

Create a local `.env` file in the project root. Do not commit this file or share API keys.

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.5-flash-lite

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=finance-mapping-agent
```

A Gemini API key is required for the LLM-assisted agent workflow. LangSmith configuration is optional and is used for tracing when enabled.

The `.env` file is excluded through `.gitignore`.

## 6. Run the Prototype

### Step 1: Profile input data

```powershell
python scripts\profile_inputs.py
```

Generates a data-quality report under `output/`.

### Step 2: Run deterministic mapping

```powershell
python scripts\run_exact_mapper.py
```

Generates:

- `output/mapping_decisions.json`
- `output/mapping_decisions.csv`

### Step 3: Run hybrid candidate retrieval

```powershell
python scripts\run_candidate_retrieval.py
```

Generates:

- `output/mapping_candidates.json`

The first run may download the embedding model from Hugging Face and build the local COA index.

### Step 4: Evaluate retrieval quality

```powershell
python scripts\evaluate_retrieval.py
```

Generates retrieval evaluation results under `output/`.

### Step 5: Run the human-review CLI

```powershell
python scripts\run_agent_review.py
```

This runs the agent workflow and allows review decisions to be submitted through the terminal.

### Step 6: Run the FastAPI application

```powershell
python -m uvicorn finance_mapper.api:app --app-dir src --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

The API includes:

- `GET /` — basic service information
- `GET /api/health` — health check
- `POST /api/run` — start a mapping-review workflow
- `POST /api/review` — submit human-review decisions and resume the workflow
- `GET /api/latest` — retrieve the latest saved report

Use the interactive API documentation to inspect the request and response schemas.

### Run the tests

From the project root, with the virtual environment activated and `PYTHONPATH` configured:

```powershell
python -m pytest -q
```

## 7. Agent Workflow

The implemented workflow follows these stages:

1. **Retrieve candidates:** Finds relevant COA accounts for unresolved source accounts using the hybrid retrieval system.
2. **Recommend:** Uses Gemini to produce a structured mapping recommendation based on the retrieved candidates.
3. **Validate and review:** Validates the recommendation against the retrieved candidate set and pauses the workflow for human review.
4. **Finalize:** Records the review decision and writes the results to `output/agent_review_results.json`.

A reviewer can approve an eligible candidate, reject a recommendation, or quarantine an account when the evidence is insufficient. Approvals are restricted to retrieved candidates by the review-validation logic.

The workflow does not create purchase orders, post journal entries, or automatically generate financial statements.

## 8. Retrieval Evaluation

The evaluation script reports:

- **Precision@3:** The fraction of the top-three returned candidates labelled relevant.
- **Recall@3:** The fraction of labelled relevant candidates retrieved in the top three.
- **MRR@3:** The mean reciprocal rank of the first relevant candidate across evaluable queries.

The current provisional benchmark contains two ranking-evaluable queries. Account `9999` is excluded from ranking metrics because no verified target mapping is available.

| Metric | Observed result |
|---|---:|
| Precision@3 | 0.3333 |
| Recall@3 | 1.0000 |
| MRR@3 | 1.0000 |
| Candidate safety rate | 100% |

These are observed prototype results, not evidence of production mapping accuracy. The dataset is very small, the reference labels require independent finance review, and the benchmark must be expanded before broader conclusions can be drawn.

Retrieval similarity and ranking scores are not calibrated probabilities or proof that an account mapping is correct.

## 9. Safety and Accounting Controls

- Preserve the original supplied input files.
- Do not automatically map unknown accounts using similarity scores alone.
- Treat retrieved accounts as candidates, not approved mappings.
- Validate LLM recommendations against retrieved candidates.
- Route ambiguous or unsupported mappings for human review.
- Do not silently repair unbalanced journal entries.
- Do not substitute an average FX rate when a required period-end rate is missing.
- Preserve duplicate source rows until their accounting meaning is confirmed.
- Do not treat balanced totals alone as proof of accounting correctness.
- Keep API credentials outside source control.

## 10. Scope and Limitations

This is a local prototype of one TB-to-COA mapping slice, not a complete financial reporting platform.

The current implementation includes deterministic mapping, hybrid retrieval, LLM-assisted recommendations, a LangGraph human-review workflow, FastAPI endpoints, and optional LangSmith tracing.

Key limitations:

- The retrieval benchmark contains only two ranking-evaluable queries.
- Human-reviewed reference labels still require independent finance validation.
- LangGraph uses an in-memory checkpointer. Workflow state is not designed to survive an application-process restart.
- Review results are written to a local JSON report; there is no production-grade persistent audit database.
- The API provides a minimal review workflow rather than a production frontend.
- RAGAS evaluation has not been implemented.
- The prototype does not generate complete financial statements or perform end-to-end ERP integration.
- It does not automatically post accounting entries or replace finance approval.

## 11. Future Improvements

- Expand the independently finance-reviewed evaluation dataset.
- Compare dense-only and hybrid retrieval on the same benchmark.
- Measure LLM recommendation quality, groundedness, and abstention behaviour.
- Add persistent workflow checkpoints and an auditable review database.
- Add role-based access controls and production security measures.
- Implement financial statement generation, FX translation, consolidation, and reconciliation as separate validated slices.
- Add integration tests and production monitoring.

## 12. Use of AI Tools

AI assistance was used for design exploration, code scaffolding, debugging, test development, and documentation. The implementation was run locally, and its observed behaviour and test results were used to verify the prototype.

LLM-generated recommendations are not treated as authoritative accounting decisions. Financial mappings remain subject to validation, appropriate evidence, and finance review.
