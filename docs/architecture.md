# Architecture: Agentic Financial Reporting Platform

**Assignment:** AI Agentic Engineer Take-Home Assignment
**Selected prototype slice:** Trial Balance (TB) to Chart of Accounts (COA) mapping

## 1. Objective and Scope

The goal is to prototype one carefully scoped component of a larger agentic financial reporting platform. The selected component maps trial balance accounts to eligible chart-of-accounts records while identifying data-quality issues and escalating uncertain cases for human review.

The prototype combines deterministic mapping, hybrid retrieval, structured LLM recommendations, and a human-in-the-loop workflow. It does not attempt to implement the entire financial reporting lifecycle.

## 2. High-Level Architecture

The processing flow is:

1. Load the supplied financial input files.
2. Profile the inputs for data-quality issues.
3. Apply deterministic account-code mapping.
4. Retrieve candidate COA records for unresolved accounts.
5. Generate a structured Gemini recommendation using the retrieved candidates.
6. Validate the recommendation against the eligible candidate set.
7. Pause the LangGraph workflow for human review.
8. Record the decision and generate a JSON report.
9. Expose the workflow through FastAPI and use LangSmith for execution tracing when configured.

## 3. Input Data and Profiling

The prototype uses the supplied files:

- `trial_balance.csv`
- `chart_of_accounts.csv`
- `prior_period_tb.csv`
- `fx_rates.csv`
- `manual_adjustments.json`

The profiling component identifies potential issues such as duplicate account-and-currency records, accounts absent from the current COA, incomplete FX-rate coverage, unbalanced manual journal entries, and ambiguous COA structures.

Profiling findings are reported rather than silently corrected. Mixed-currency amounts must not be treated as directly comparable without appropriate currency handling.

## 4. Deterministic Mapping

The deterministic mapper compares source account codes with eligible COA records.

It produces mapping decisions with statuses such as:

- `exact_match`
- `needs_review`
- `unmapped`

The mapper handles potentially ambiguous or structurally unsuitable COA records explicitly. Exact matching is used where appropriate; an account that cannot be mapped safely is not automatically assigned to a similar account.

This layer provides a transparent baseline before retrieval or LLM recommendations are considered.

## 5. Hybrid Candidate Retrieval

For unresolved accounts, the retrieval component identifies possible COA targets.

### Dense retrieval

Sentence Transformers generates embeddings using `all-MiniLM-L6-v2`. ChromaDB stores and retrieves account representations using cosine-distance-based search.

### Keyword retrieval

BM25 ranks candidates based on lexical overlap between the source account information and COA descriptions.

### Hybrid ranking

Reciprocal Rank Fusion (RRF) combines the dense and keyword rankings. The two approaches complement one another: semantic retrieval can help with related terminology, while keyword retrieval can preserve useful lexical matches.

COA records are represented at account level rather than split into arbitrary text chunks. Header accounts and accounts that have children are excluded as mapping targets.

The resulting candidates provide supporting evidence for recommendations. Retrieval scores are ranking signals, not calibrated probabilities that a mapping is correct.

## 6. LLM-Assisted Recommendation

The recommendation node uses Gemini through `langchain-google-genai`. A Pydantic schema is used to structure the model output.

The model considers the unresolved account and the retrieved candidates to recommend a mapping when the available evidence is sufficient. It can also abstain when the candidates do not support a defensible recommendation.

A recommendation is not treated as an authoritative accounting decision. The implementation validates that the recommended target belongs to the retrieved candidate set before the workflow proceeds.

## 7. LangGraph and Human-in-the-Loop Workflow

The agent workflow consists of four primary nodes:

1. **`retrieve_candidates`** — retrieves candidate COA accounts.
2. **`recommend_candidates`** — requests a structured recommendation from Gemini.
3. **`human_review`** — interrupts the workflow to request a human decision.
4. **`finalize`** — records the decisions and writes the final report.

The review process supports approve, reject, and quarantine actions. Approval is restricted to eligible retrieved candidates; a reviewer cannot approve an arbitrary account outside that candidate set.

LangGraph uses an `InMemorySaver` checkpointer to maintain workflow state during the running application process. The FastAPI integration starts the workflow and resumes it after a review submission.

This is an implemented prototype review flow, not a production-grade approval system. The in-memory checkpoint does not provide durable workflow recovery across application restarts.

## 8. FastAPI and Observability

FastAPI exposes the workflow through these endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | Basic service information |
| `GET /api/health` | Health check |
| `POST /api/run` | Start a mapping-review workflow |
| `POST /api/review` | Submit review decisions and resume the workflow |
| `GET /api/latest` | Retrieve the latest saved report |

LangSmith can trace workflow execution and model calls when the environment variables are configured. This supports inspection and debugging of the prototype; it does not by itself provide complete production monitoring or governance.

## 9. Outputs and Evaluation

The prototype writes machine-readable reports for data-quality profiling, deterministic mapping, candidate retrieval, retrieval evaluation, and completed agent-review results.

Retrieval evaluation reports Precision@3, Recall@3, and MRR@3 against a small provisional reference dataset. The current benchmark contains two ranking-evaluable queries; account `9999` is excluded from ranking metrics because a verified target mapping is unavailable.

The observed results are:

| Metric | Result |
|---|---:|
| Precision@3 | 0.3333 |
| Recall@3 | 1.0000 |
| MRR@3 | 1.0000 |
| Candidate safety rate | 100% |

These results demonstrate a limited prototype evaluation, not production accuracy. The reference labels require independent finance review, and the benchmark must be expanded to support meaningful conclusions.

## 10. Controls and Failure Handling

The implementation is designed around these principles:

- Preserve supplied input data rather than silently changing it.
- Flag unknown, ambiguous, or structurally unsuitable accounts.
- Treat source files and retrieved content as data, not instructions to the model.
- Restrict LLM recommendations and human approvals to the eligible candidate set.
- Allow unresolved cases to be rejected or quarantined.
- Avoid using similarity scores as proof of accounting correctness.
- Keep journal-entry validation and FX requirements distinct from account mapping.

## 11. Limitations and Future Work

This prototype does not generate complete financial statements, post accounting entries, or perform end-to-end ERP integration.

It does not yet include a persistent audit database, durable workflow checkpoints, role-based access control, a production frontend, or RAGAS evaluation. The evaluation dataset is small, and the review API is intended for demonstration rather than production use.

Potential next steps include expanding the finance-reviewed benchmark, evaluating recommendation groundedness and abstention, adding durable workflow state and review auditing, strengthening API security, and implementing additional validated slices for FX translation, consolidation, reconciliation, and financial statement generation.
