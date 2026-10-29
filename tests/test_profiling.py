from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finance_mapper.profiling import profile_inputs  # noqa: E402

INPUTS = PROJECT_ROOT / "inputs"


def finding_codes(report):
    return {item["code"] for item in report["findings"]}


def test_profiles_all_input_files():
    report = profile_inputs(INPUTS)
    assert report["files"]["trial_balance_rows"] == 62
    assert report["files"]["chart_of_accounts_rows"] == 72
    assert report["files"]["prior_period_rows"] == 33
    assert report["files"]["fx_rate_rows"] == 7
    assert report["files"]["adjustment_entries"] == 10


def test_flags_known_seeded_issues_without_cleaning_inputs():
    report = profile_inputs(INPUTS)
    codes = finding_codes(report)
    assert "TB_ACCOUNT_NOT_IN_COA" in codes
    assert "DUPLICATE_TB_ACCOUNT_CURRENCY" in codes
    assert "COA_HEADER_WITHOUT_CHILDREN" in codes
    assert "COA_NON_HEADER_WITH_CHILDREN" in codes
    assert "COA_CF_CATEGORY_TBD" in codes
    assert "FX_PERIOD_END_RATE_MISSING" in codes
    assert "PRIOR_ACCOUNT_NOT_IN_CURRENT_COA" in codes
    assert "ADJUSTMENT_OUT_OF_BALANCE" in codes
    assert "ADJUSTMENT_ACCOUNT_NOT_IN_COA" in codes
    assert "ADJUSTMENT_REPEATED_ACCOUNT_LINES" in codes


def test_profiler_does_not_mutate_source_files():
    before = (INPUTS / "trial_balance.csv").read_bytes()
    profile_inputs(INPUTS)
    after = (INPUTS / "trial_balance.csv").read_bytes()
    assert before == after
