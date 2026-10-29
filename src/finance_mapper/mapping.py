from pathlib import Path

import pandas as pd

from .schemas import MappingDecision


def map_trial_balance_accounts(
    input_dir: Path,
) -> list[MappingDecision]:
    """Map TB account codes to unambiguous COA posting accounts."""

    tb = pd.read_csv(
        input_dir / "trial_balance.csv",
        dtype={"account_code": "string"},
    )

    coa = pd.read_csv(
    input_dir / "chart_of_accounts.csv",
    dtype={
        "account_code": "string",
        "parent_code": "string",
    },
)

    required_tb = {"account_code", "account_name"}
    required_coa = {"account_code", "account_name", "account_type"}

    if not required_tb.issubset(tb.columns):
        raise ValueError("Trial balance is missing required columns.")

    if not required_coa.issubset(coa.columns):
        raise ValueError("COA is missing required columns.")

    # Normalize codes for comparison without changing source files.
    tb["account_code"] = tb["account_code"].str.strip()
    coa["account_code"] = coa["account_code"].str.strip()
    # Collect account codes that have child accounts in the COA.
    coa_parent_codes = set(
    coa["parent_code"].dropna().astype(str).str.strip()
)

    # One mapping decision per distinct source account identity.
    accounts = tb[
        ["account_code", "account_name"]
    ].drop_duplicates()

    decisions: list[MappingDecision] = []

    for row in accounts.itertuples(index=False):
        source_code = (
            str(row.account_code).strip()
            if pd.notna(row.account_code)
            else ""
        )
        source_name = (
            str(row.account_name).strip()
            if pd.notna(row.account_name)
            else None
        )

        matches = coa.loc[
            coa["account_code"].eq(source_code)
            & coa["account_code"].notna()
        ]


        # Case 1: No code to map, or code doesn't exist in the COA.
        if not source_code or matches.empty:
            reason = (
                "Source account code is missing."
                if not source_code
                else "No exact account-code match exists in the current COA."
            )

            decisions.append(
                MappingDecision(
                    source_account_code=source_code,
                    source_account_name=source_name,
                    target_account_code=None,
                    target_account_name=None,
                    status="unmapped",
                    method="none",
                    confidence_score=0.0,
                    reason=reason,
                    evidence=[
                        f"Source account name: {source_name or 'Missing'}",
                        "A candidate mapping requires additional evidence.",
                    ],
                    requires_human_review=True,
                )
            )
            continue

        # Case 2: Multiple COA entries share the same code.
        if len(matches) > 1:
            decisions.append(
                MappingDecision(
                    source_account_code=source_code,
                    source_account_name=source_name,
                    target_account_code=None,
                    target_account_name=None,
                    status="needs_review",
                    method="none",
                    confidence_score=0.0,
                    reason="Multiple COA records share this account code.",
                    evidence=[
                        f"Candidate COA record: {r.account_name}"
                        for r in matches.itertuples(index=False)
                    ],
                    requires_human_review=True,
                )
            )
            continue

        target = matches.iloc[0]
        target_code = str(target["account_code"]).strip()
        target_name = str(target["account_name"]).strip()
        account_type = str(target["account_type"]).strip()
        cf_category = (
    str(target["cf_category"]).strip()
    if pd.notna(target["cf_category"])
    else None
)

        # Case: Account has children, so posting/roll-up
# behavior needs human review.
        if target_code in coa_parent_codes:
            decisions.append(
        MappingDecision(
            source_account_code=source_code,
            source_account_name=source_name,
            target_account_code=None,
            target_account_name=None,
            status="needs_review",
            method="exact_code",
            confidence_score=0.0,
            reason=(
                "The COA account has child accounts; "
                "posting and roll-up semantics require review."
            ),
            evidence=[
                f"COA code: {target_code}",
                f"COA name: {target_name}",
                f"COA account type: {account_type}",
            ],
            requires_human_review=True,
        )
    )
            continue

        # Case 3: A header is not a valid posting-account target.
        if account_type.casefold() == "header":
            decisions.append(
                MappingDecision(
                    source_account_code=source_code,
                    source_account_name=source_name,
                    target_account_code=None,
                    target_account_name=None,
                    status="needs_review",
                    method="exact_code",
                    confidence_score=0.0,
                    reason="The matching COA node is a header, not a posting account.",
                    evidence=[
                        f"COA code: {target_code}",
                        f"COA name: {target_name}",
                        f"COA account type: {account_type}",
                    ],
                    requires_human_review=True,
                )
            )
            continue

        # Case 4: Code matches, but the descriptions conflict.
        names_agree = (
            source_name is not None
            and source_name.casefold() == target_name.casefold()
        )

        if not names_agree:
            decisions.append(
                MappingDecision(
                    source_account_code=source_code,
                    source_account_name=source_name,
                    target_account_code=target_code,
                    target_account_name=target_name,
                    status="needs_review",
                    method="exact_code",
                    confidence_score=0.70,
                    reason=(
                        "The account code matches, but the source and COA "
                        "names differ or the source name is missing."
                    ),
                    evidence=[
                        f"Trial balance name: {source_name or 'Missing'}",
                        f"COA name: {target_name}",
                        f"COA account type: {account_type}",
                    ],
                    requires_human_review=True,
                )
            )
            continue
        if cf_category is not None and cf_category.casefold() == "tbd":
            decisions.append(
        MappingDecision(
            source_account_code=source_code,
            source_account_name=source_name,
            target_account_code=target_code,
            target_account_name=target_name,
            status="needs_review",
            method="exact_code",
            confidence_score=1.0,
            reason=(
                "The account code and name match, but the "
                "cash-flow category is TBD."
            ),
            evidence=[
                f"COA code: {target_code}",
                f"COA name: {target_name}",
                f"Cash-flow category: {cf_category}",
            ],
            requires_human_review=True,
        )
    )
            continue

        # Case 5: A unique, valid, exact match.
        decisions.append(
            MappingDecision(
                source_account_code=source_code,
                source_account_name=source_name,
                target_account_code=target_code,
                target_account_name=target_name,
                status="exact_match",
                method="exact_code",
                confidence_score=1.0,
                reason="Unique account-code match with an agreeing account name.",
                evidence=[
                    f"Exact code match: {source_code}",
                    f"Matching COA name: {target_name}",
                    f"COA account type: {account_type}",
                ],
                requires_human_review=False,
            )
        )

    return decisions
