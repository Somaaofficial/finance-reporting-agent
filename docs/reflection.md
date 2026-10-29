# Project Reflection: TB-to-COA Mapping Prototype

## Objective

I selected Trial Balance to Chart of Accounts mapping as the prototype slice because mapping errors can affect downstream financial reporting. Rather than attempting to build a complete multi-agent financial reporting platform, I focused on creating a workflow that combines deterministic processing, retrieval, an LLM recommendation, and human review.

## Key Design Decisions

I implemented deterministic mapping first because exact account-code matches are easier to explain and validate than model-generated decisions. Data-quality profiling was kept separate so that issues such as duplicate records, missing FX rates, and unbalanced adjustments are identified rather than silently repaired.

For unresolved accounts, I implemented hybrid retrieval using Sentence Transformers, ChromaDB, BM25, and Reciprocal Rank Fusion. Dense retrieval helps surface semantically related descriptions, while BM25 contributes lexical matching. RRF combines their rankings without directly treating their differently scaled scores as equivalent. COA accounts are indexed as individual records rather than arbitrary text chunks.

I then added a Gemini recommendation node with structured Pydantic output and candidate validation. LangGraph coordinates retrieval, recommendation, human review, and finalization. The workflow can pause for a review decision and resume after the reviewer submits an action. FastAPI exposes the workflow, and LangSmith tracing supports inspection of execution and model calls.

## Evaluation and Observations

The current provisional retrieval benchmark reports Precision@3 of 0.3333, Recall@3 of 1.0000, MRR@3 of 1.0000, and a candidate safety rate of 100%.

These results must be interpreted cautiously because only two queries currently have provisional relevance judgments. A perfect recall or candidate-safety result on such a small dataset does not demonstrate production-level reliability. The labels need independent finance review, and the benchmark needs more representative accounts and edge cases.

The workflow also showed why the system should be able to abstain. An account with insufficient evidence should not receive an automatic mapping simply because a candidate has a high retrieval score or an LLM produces a plausible answer.

## Limitations and Lessons Learned

The implementation is a local prototype rather than a production accounting system. LangGraph uses an in-memory checkpointer, so workflow state is not durable across application restarts. Review results are written to local JSON files rather than a persistent audit database. There is no production frontend, complete access-control model, RAGAS evaluation, ERP posting, or full financial statement generation.

A central lesson was that using an LLM does not remove the need for deterministic validation. Structured output helps constrain the response format, but it does not guarantee that the recommendation is correct. Restricting recommendations to retrieved candidates and requiring human review for unresolved mappings reduces some risks without eliminating them.

I also learned that evaluation quality depends on the quality and size of the reference dataset. Retrieval metrics are useful for comparing approaches, but the benchmark must reflect real mapping scenarios before the results can support operational decisions.

## Use of AI Tools

AI assistance was used for design exploration, code scaffolding, debugging, test development, and documentation. I ran the implementation locally and used its observed behaviour and test results to verify the prototype. The system's generated recommendations are treated as suggestions, not authoritative accounting decisions. Finance review and independent validation remain necessary.
