from pathlib import Path

from finance_mapper.retrieval import retrieve_coa_candidates


ROOT = Path(__file__).resolve().parents[1]

report = retrieve_coa_candidates(
    input_dir=ROOT / "inputs",
    output_dir=ROOT / "output",
    top_k=3,
)

print("Candidate retrieval completed.")
print("COA accounts indexed:", report["candidate_accounts_indexed"])
print("Unresolved source accounts:", report["unresolved_source_accounts"])
print("Candidates returned:", report["candidate_count"])
print("Latency:", report["latency_seconds"])
print("Report:", ROOT / "output" / "mapping_candidates.json")

for candidate in report["candidates"]:
    print(
        f"{candidate['source_account_code']} -> "
        f"{candidate['candidate_account_code']} "
        f"({candidate['similarity_score']:.4f})"
    )