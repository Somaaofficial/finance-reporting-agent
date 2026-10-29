import json
from collections import Counter
from pathlib import Path

import pandas as pd

from finance_mapper.mapping import map_trial_balance_accounts


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "output"


def main():
    decisions = map_trial_balance_accounts(INPUT_DIR)

    # Convert validated Pydantic models to JSON-compatible dictionaries.
    records = [
        decision.model_dump(mode="json")
        for decision in decisions
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "mapping_decisions.json"
    csv_path = OUTPUT_DIR / "mapping_decisions.csv"

    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Keep evidence readable in the CSV output.
    csv_records = []
    for record in records:
        row = record.copy()
        row["evidence"] = " | ".join(row["evidence"])
        csv_records.append(row)

    pd.DataFrame(csv_records).to_csv(csv_path, index=False)

    status_counts = Counter(
        decision.status for decision in decisions
    )

    print("Exact account mapping completed.")
    print("Distinct account identities:", len(decisions))
    print("Decisions by status:", dict(status_counts))
    print("JSON report:", json_path)
    print("CSV report:", csv_path)


if __name__ == "__main__":
    main()