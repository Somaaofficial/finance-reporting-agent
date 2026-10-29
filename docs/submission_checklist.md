# Submission Checklist — AI Agentic Engineer Take-Home

## Before submission

- [ ] Confirm `requirements.txt` includes every dependency used by the scripts, including `rank-bm25`.
- [ ] Confirm `.gitignore` excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, `.env`, and local secrets.
- [ ] Install from `requirements.txt` in a clean environment if time permits.
- [ ] Run `python -m pytest -q` and keep the final passing result.
- [ ] Run `python scripts\profile_inputs.py`.
- [ ] Run `python scripts\run_exact_mapper.py`.
- [ ] Run `python scripts\run_candidate_retrieval.py`.
- [ ] Run `python scripts\evaluate_retrieval.py`.
- [ ] Check that the expected reports exist in `output/`.
- [ ] Verify that supplied files in `inputs/` remain unchanged.
- [ ] Ensure `docs/architecture.md` and `docs/reflection.md` accurately describe the implemented prototype and its limitations.
- [ ] Do not commit virtual environments, API keys, tokens, or personal data.
- [ ] Commit and push the repository to GitHub; verify the repository URL and access settings.

## Main commands (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
python scripts\profile_inputs.py
python scripts\run_exact_mapper.py
python scripts\run_candidate_retrieval.py
python scripts\evaluate_retrieval.py
python -m pytest -q
```

## Submission summary

- **Implemented slice:** Trial Balance to Chart of Accounts mapping with deterministic decisions and review-required candidates.
- **Retrieval:** Sentence Transformers + ChromaDB cosine search, BM25 keyword ranking, and Reciprocal Rank Fusion.
- **Evaluation:** Precision@3, Recall@3 and MRR@3 on a tiny provisional benchmark; clearly disclose its limitations.
- **Not implemented:** complete statement generation, an LLM recommendation node, RAGAS evaluation, a review UI/API, or persistent human approval workflow.
