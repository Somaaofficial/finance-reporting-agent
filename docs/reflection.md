# One-Page Reflection

## What I would build differently with three months

With three months, I would first work with controllership and consolidation specialists to define canonical accounting semantics and create an independently reviewed gold dataset for account mappings, FX treatment, journal-adjustment dispositions, and audit lineage. I would then add versioned and effective-dated mappings, a persistent review ledger, and repeatable evaluation. After agreeing on stable input contracts, I would integrate ERP connectors. The four financial statements would be separate deterministic builders that consume an approved canonical ledger and share a validation layer.

## Where the prototype breaks at scale

The current prototype uses a small, local dataset and local retrieval index. At thousands of accounts or multiple entities, it would need batching, version-aware indexing, retries, idempotency, concurrency controls, and persistent run metadata. Intercompany elimination cannot be inferred from account codes or descriptions alone; entity, counterparty, currency, and reciprocal transaction references are needed. FX translation depends on approved accounting policy and rate dates, and a missing critical rate must block the affected calculation rather than be silently replaced. Reviewer decisions also need authorization, separation of duties, and a durable audit trail linked to the COA version and reporting period.

## How AI tools helped—and where judgment was still needed

I used ChatGPT as a pair-programming and analysis assistant to interpret the brief, scaffold code and tests, troubleshoot errors, and refine documentation. I then ran the scripts and tests locally and used observed results to validate the current behavior. I learned that a plausible retrieval result is not automatically a correct accounting mapping. I also tested text-preparation changes and observed that changing the document representation affected ranking, which reinforced the need to measure retrieval changes rather than assume that a cleaner representation is always better. The current retrieval benchmark is very small and its relevance labels are provisional, so I do not claim production accuracy. No LLM-generated accounting recommendation or approval workflow is currently implemented.

## One underestimated challenge

The difficult part is not simply selecting an embedding model or adding an LLM. It is defining accounting semantics, versioning, evidence, and ownership of review decisions. A semantically plausible label can still be wrong when normal balance, financial-statement treatment, cash-flow classification, entity, or effective date differs. Without a finance-approved policy and reliable lineage, a fluent explanation can make an unsafe result appear trustworthy. The system should expose uncertainty and request evidence rather than hide it.
