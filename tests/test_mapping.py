from collections import Counter
from pathlib import Path

from finance_mapper.mapping import map_trial_balance_accounts


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "inputs"


def get_decision(decisions, account_code):
    matches = [
        decision
        for decision in decisions
        if decision.source_account_code == account_code
    ]

    assert len(matches) == 1, (
        f"Expected one decision for account {account_code}, "
        f"found {len(matches)}"
    )

    return matches[0]


def test_mapping_status_counts():
    decisions = map_trial_balance_accounts(INPUT_DIR)

    counts = Counter(
        decision.status for decision in decisions
    )

    assert len(decisions) == 59
    assert counts == {
        "exact_match": 56,
        "needs_review": 2,
        "unmapped": 1,
    }


def test_valid_exact_match():
    decisions = map_trial_balance_accounts(INPUT_DIR)

    decision = get_decision(decisions, "6310")

    assert decision.status == "exact_match"
    assert decision.target_account_code == "6310"
    assert decision.requires_human_review is False


def test_tbd_classifications_require_review():
    decisions = map_trial_balance_accounts(INPUT_DIR)

    for account_code in ["1150", "2170"]:
        decision = get_decision(decisions, account_code)

        assert decision.status == "needs_review"
        assert decision.requires_human_review is True


def test_unmapped_account_is_not_assigned():
    decisions = map_trial_balance_accounts(INPUT_DIR)

    decision = get_decision(decisions, "9999")

    assert decision.status == "unmapped"
    assert decision.target_account_code is None
    assert decision.requires_human_review is True