# Architecture: Financial Reporting Agent — Trial Balance to COA Mapping

**Assignment:** AI Agentic Engineer take-home
**Selected prototype slice:** Trial Balance (TB) to Chart of Accounts (COA) mapping
**Implementation status:** Local Python prototype for read-only profiling, deterministic mapping, hybrid retrieval, and retrieval evaluation. It is not a complete financial reporting system.

## 1. Problem framing and scope

The assignment describes a broader platform that would ingest ERP trial balances, a COA, manual adjustments, prior-period data, and FX rates, then generate the Balance Sheet, Profit and Loss statement, Cash Flow Statement, and Statement of Changes in Equity. The work here intentionally implements one bounded slice rather than shallowly attempting all four statements: identify direct account mappings and surface candidates for unresolved accounts without silently approving uncertain accounting decisions.

The current prototype does **not** generate the four financial statements, call an LLM, provide a web/API review interface, persist reviewer decisions, or post accounting entries. Those are possible future extensions, not implemented capabilities.

## 2. Implemented workflow

```text
Supplied CSV/JSON inputs (preserved)
              |
              v
Read-only input profiling
              |
              +----> output/data_quality_report.json
              |
              v
Deterministic TB-to-COA mapping
              |
              +----> output/mapping_decisions.json / .csv
              |
              v
Collect unresolved accounts from TB, prior TB and adjustments
              |
              v
Build one searchable document per eligible COA posting account
              |
              +----------------------+---------------------+
              |                                            |
              v                                            v
SentenceTransformer embeddings                    BM25 keyword retrieval
              |                                            |
              v                                            |
          Chroma cosine ranking                          |
              |                                            |
              +----------------------+---------------------+
                                     v
                       Reciprocal Rank Fusion (RRF)
                                     |
                                     v
                     Top-3 candidate records per query
                                     |
                                     v
                    output/mapping_candidates.json
                                     |
                                     v
                    Offline retrieval evaluation
                                     |
                                     v
                    output/retrieval_evaluation.json
```

### Implemented components

1. **Input profiling (`scripts/profile_inputs.py`)** — reads the supplied financial inputs and writes a data-quality report. The observed run found 11 items: 5 high severity, 5 medium severity, and 1 informational finding. Findings are reported, not silently fixed.
2. **Deterministic mapping (`scripts/run_exact_mapper.py`)** — checks source account identities against eligible COA records. The latest run processed 59 distinct account identities: 56 exact matches, 2 review-required decisions, and 1 unmapped account.
3. **Candidate retrieval (`scripts/run_candidate_retrieval.py`)** — identifies unresolved accounts from the current TB, prior-period TB, and manual adjustment references. It indexed 58 eligible COA accounts and returned three candidates for each of three unresolved source accounts.
4. **Dense retrieval** — `sentence-transformers/all-MiniLM-L6-v2` embeds account descriptions and source queries. Embeddings are normalized and stored in a local Chroma collection configured with cosine distance. The reported semantic similarity is derived from cosine distance as `1 - distance`; it is a ranking signal, not a calibrated probability.
5. **Keyword retrieval and fusion** — `rank-bm25` ranks account documents lexically. Reciprocal Rank Fusion (RRF) combines the dense and BM25 ranks. The candidate output retains cosine similarity as a separate field and records BM25 rank and RRF score in its evidence.
6. **Retrieval evaluation (`scripts/evaluate_retrieval.py`)** — calculates Precision@3, Recall@3, MRR@3, and a candidate-flag safety rate against a small provisional reference set.

## 3. Deterministic rules versus AI-assisted capabilities

### Deterministic code is responsible for

- Preserving provided input files and identifying data-quality issues.
- Comparing source account codes to COA codes and identifying review/unmapped conditions.
- Excluding COA headers and accounts with children from eligible mapping targets.
- Generating embeddings, running vector search, BM25 retrieval, and fusing rankings.
- Calculating evaluation metrics from explicitly supplied relevance labels.
- Keeping retrieved results in candidate-only / human-review-required states rather than treating similarity as approval.

### Current AI-related capability

The implemented model use is embedding-based semantic retrieval via Sentence Transformers. The current prototype does **not** make an LLM call to explain or choose a mapping. BM25, vector search, and RRF are retrieval components, not an LLM reasoning agent.

### Proposed future LLM capability (not implemented)

A future optional LLM node could receive only a source description and a bounded list of retrieved COA candidates, then return a structured recommendation or abstain. Its proposed target would need deterministic validation against the allowed posting-account candidate list. It must never invent an account code, alter input data, or approve its own accounting decision. Any external model call would require approved handling of financial data and measured token/latency telemetry.

## 4. Findings from the supplied inputs and safe handling

| Finding | Observed issue | Safe handling |
|---|---|---|
| Orphan current account | `9999 — Suspense - Unmapped` is not in the current COA | Keep unmapped/quarantined; do not select a candidate solely because it ranks highest. |
| Prior-period account | `6905 — Sundry Operating Expenses` is absent from the current COA; `6900 — Other Operating Expenses` is a plausible candidate | Surface as a review-required candidate; require finance approval and period/COA-version confirmation. |
| Adjustment account missing from COA | JE-005 references `6315` with a conference-travel memo | `6310 — Travel and Entertainment` is a relevant candidate, but remains unapproved pending review. |
| Missing period-end FX rate | GBP appears in the TB but has no period-end rate in the supplied FX file | Do not silently substitute an average rate; block any downstream calculation requiring the missing closing rate. |
| Unbalanced adjustment | JE-002 has a debit/credit difference of USD 3,500 | Quarantine the proposed entry and request corrected evidence; do not add a balancing plug. |
| Intercompany semantics unclear | JE-008 debits and credits account 2170 for USD 320,000, but the lines do not supply sufficient entity/counterparty context | Require finance review; equal debit/credit totals alone do not prove semantic validity. |
| COA category unresolved | Accounts 1150 and 2170 have `TBD` cash-flow categories | Flag the category for review rather than infer it from the account name. |
| COA hierarchy issues | Header 1290 has no children; account 8000 is marked non-header but has children | Report for review before implementing statement roll-ups to avoid missing or double-counting balances. |
| Duplicate account/currency rows | Account 6310 appears more than once in USD | Preserve both rows and investigate whether they are split postings or duplicate exports. |

The supplied trial balance contains multiple currencies. Raw amounts from different currencies must not be added together to establish a functional-currency balance. FX policy and rate type must be agreed before statement translation or reconciliation.

## 5. Retrieval evaluation and interpretation

The current evaluation set has only two ranking-evaluable queries. `9999` is excluded from ranking metrics because there is no verified target mapping for that generic suspense description. The provisional relevant candidates are `6900` for `6905` and `6310` for `6315`; these labels need independent finance validation.

Latest observed hybrid run:

| Metric | Result |
|---|---:|
| Precision@3 | 0.3333 |
| Recall@3 | 1.0000 |
| MRR@3 | 1.0000 |
| Candidate flag safety rate | 100% |

Dense-only baseline and hybrid retrieval produced the same reported metrics on this tiny benchmark. BM25 changed some lower candidate ranks, but no overall improvement was demonstrated. The current benchmark is a smoke test, not an unbiased holdout or evidence of production accuracy. The “candidate safety rate” checks output flags only; it does not mean a real human approval workflow exists.

A stronger evaluation should use a larger, independently finance-reviewed gold set. It should measure top-1/top-k relevance, abstention behavior for accounts with insufficient evidence, reviewer overturn rate when review functionality exists, and latency on repeated runs. Similarity values should not be presented as mapping probabilities.

## 6. Auditability and limitations

The current artifacts preserve source account codes, source account names when available, source file/adjustment references, candidate COA codes/names, retrieval evidence, rank information, and review-required flags. They do not yet implement complete journal-line-to-statement-cell lineage because statement generation is outside the implemented slice.

The current prototype is a local scripted workflow. It does not include a deployed API, persistent approval/rejection workflow, LangGraph orchestration, LLM recommendation node, RAGAS evaluation, or statement generation. These should not be described as implemented in the README or demonstration.

## 7. Recommended next increments

1. Increase the independently reviewed retrieval evaluation set and compare dense-only with hybrid retrieval on the same cases.
2. Add structured LLM recommendations only if evidence supports the need; enforce a candidate allowlist, Pydantic validation, explicit abstention, and human approval.
3. Add a persistent review ledger with reviewer identity, decision, rationale, timestamp, run ID, and COA version.
4. Add API/UI workflows only after the input/output contracts and approval process are agreed.
5. Implement deterministic adjustment validation, FX translation, consolidation, statement construction, and reconciliation as separately testable modules in a future full system.

## 8. Clarifying questions for finance owners

1. What are the approved FX rate types by statement and account class, and how must a missing GBP closing rate be resolved?
2. Who is authorized to approve non-exact account mappings and unresolved cash-flow classifications, and should all such mappings remain review-required?
3. Which entity, counterparty, and reciprocal-entry fields are required to validate intercompany journals and eliminations?
