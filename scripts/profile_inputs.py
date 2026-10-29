from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_mapper.profiling import profile_inputs, write_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only profiler for the finance assignment inputs.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "inputs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "data_quality_report.json")
    args = parser.parse_args()

    report = profile_inputs(args.input_dir)
    write_report(report, args.output)

    print("Read-only input profiling completed.")
    print(f"Input directory: {args.input_dir}")
    print(f"Report: {args.output}")
    print(f"Findings: {report['finding_count']}")
    print("Counts by severity:", report["findings_by_severity"])
    print("\nFindings summary:")
    for finding in report["findings"]:
        print(f"- [{finding['severity'].upper()}] {finding['code']}: {finding['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
