"""Read-only profiling of the supplied financial input bundle.

This module reports potential data-quality issues. It intentionally does not
repair, overwrite, or normalize the source files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

#check if there are any missing required columns in the input dataframes
def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Return JSON-friendly row records from selected columns."""
    safe = df[columns].astype("object").where(pd.notna(df[columns]), None)
    return safe.to_dict(orient="records")


def profile_inputs(input_dir: Path) -> dict[str, Any]:
    """Profile all five source files and return a structured findings report."""
    tb_path = input_dir / "trial_balance.csv"
    coa_path = input_dir / "chart_of_accounts.csv"
    prior_path = input_dir / "prior_period_tb.csv"
    fx_path = input_dir / "fx_rates.csv"
    adjustments_path = input_dir / "manual_adjustments.json"

    required_paths = [tb_path, coa_path, prior_path, fx_path, adjustments_path]
    missing_files = [p.name for p in required_paths if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required input file(s): {', '.join(missing_files)}")

    # Read codes as strings so leading zeroes or formatting are not lost.
    # We do not edit these source files; the DataFrames are read-only copies.
    tb = pd.read_csv(tb_path, dtype={"account_code": "string", "currency": "string"})
    coa = pd.read_csv(
        coa_path,
        dtype={"account_code": "string", "parent_code": "string"},
    )
    prior = pd.read_csv(
        prior_path,
        dtype={"account_code": "string", "currency": "string"},
    )
    fx = pd.read_csv(fx_path, dtype={"currency": "string", "rate_type": "string"})
    adjustments = json.loads(adjustments_path.read_text(encoding="utf-8"))

    findings: list[dict[str, Any]] = []  #default factory for findings

    def add(severity: str, code: str, message: str, evidence: Any = None) -> None:
        findings.append({
            "severity": severity,
            "code": code,
            "message": message,
            "evidence": evidence,
        })

    # Basic file statistics and required-column checks.
    expected_columns = {
        "trial_balance.csv": {"account_code", "account_name", "currency", "debit", "credit"},
        "chart_of_accounts.csv": {"account_code", "account_name", "account_type", "parent_code", "statement", "cf_category", "normal_balance"},
        "prior_period_tb.csv": {"account_code", "account_name", "currency", "debit", "credit"},
        "fx_rates.csv": {"currency", "rate_type", "rate", "period"},
    }
    frames = {
        "trial_balance.csv": tb,
        "chart_of_accounts.csv": coa,
        "prior_period_tb.csv": prior,
        "fx_rates.csv": fx,
    }
    for filename, required in expected_columns.items():
        missing = sorted(required - set(frames[filename].columns))
        if missing:
            add("error", "MISSING_COLUMNS", f"{filename} is missing required columns.", {"columns": missing})

    if any(f["code"] == "MISSING_COLUMNS" for f in findings):
        raise ValueError("Input schema check failed. See MISSING_COLUMNS findings in code output.")

    # Account code and currency integrity in the current TB.
    coa_codes = set(coa["account_code"].dropna().astype(str).str.strip())
    tb_codes = tb["account_code"].dropna().astype(str).str.strip()
    unknown_tb = tb.loc[~tb_codes.isin(coa_codes), ["account_code", "account_name", "currency", "debit", "credit"]]
    if not unknown_tb.empty:
        add("high", "TB_ACCOUNT_NOT_IN_COA", "Trial-balance accounts are missing from the current chart of accounts.", _records(unknown_tb, list(unknown_tb.columns)))

    # Duplicate account/currency rows are surfaced; they are not dropped.
    duplicate_mask = tb.duplicated(subset=["account_code", "currency"], keep=False)
    duplicate_groups = (
        tb.loc[duplicate_mask]
        .groupby(["account_code", "currency"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    if not duplicate_groups.empty:
        add("medium", "DUPLICATE_TB_ACCOUNT_CURRENCY", "Multiple trial-balance rows share an account code and currency. Investigate whether rows should be aggregated or represent duplicate postings.", _records(duplicate_groups, list(duplicate_groups.columns)))

    # Balance diagnostics: show raw totals only as diagnostics, not as a cross-currency accounting conclusion.
    tb["debit"] = pd.to_numeric(tb["debit"], errors="coerce")
    tb["credit"] = pd.to_numeric(tb["credit"], errors="coerce")
    totals_by_currency = tb.groupby("currency", dropna=False)[["debit", "credit"]].sum(min_count=1).reset_index()
    totals_by_currency["debit_minus_credit"] = totals_by_currency["debit"] - totals_by_currency["credit"]
    add("info", "TB_TOTALS_BY_CURRENCY", "Raw debit and credit totals are reported by currency. Do not add currencies together; translate using the appropriate FX policy before a functional-currency balance check.", _records(totals_by_currency, list(totals_by_currency.columns)))

    # COA quality: duplicate codes, missing parents, parent references and leaf/header concerns.
    duplicate_coa = coa.loc[coa.duplicated(subset=["account_code"], keep=False), ["account_code", "account_name", "account_type"]]
    if not duplicate_coa.empty:
        add("high", "DUPLICATE_COA_CODE", "The COA contains duplicate account codes.", _records(duplicate_coa, list(duplicate_coa.columns)))

    parent_codes = set(coa["account_code"].dropna().astype(str).str.strip())
    has_parent = coa["parent_code"].notna() & coa["parent_code"].astype("string").str.strip().ne("")
    invalid_parent = coa.loc[has_parent & ~coa["parent_code"].astype("string").str.strip().isin(parent_codes), ["account_code", "account_name", "parent_code"]]
    if not invalid_parent.empty:
        add("medium", "COA_PARENT_NOT_FOUND", "Some COA nodes reference a parent code that does not exist.", _records(invalid_parent, list(invalid_parent.columns)))

    children = set(coa["parent_code"].dropna().astype(str).str.strip())
    headers_without_children = coa.loc[
        coa["account_type"].astype("string").str.casefold().eq("header")
        & ~coa["account_code"].astype("string").str.strip().isin(children),
        ["account_code", "account_name", "parent_code", "statement"],
    ]
    if not headers_without_children.empty:
        add("medium", "COA_HEADER_WITHOUT_CHILDREN", "Header nodes with no direct children need confirmation before statement roll-up.", _records(headers_without_children, list(headers_without_children.columns)))

    # A posting account that also has children can be double-counted if both the
    # parent balance and its children are rolled up into statements. Flag it; do
    # not assume which level is authoritative without accounting rules.
    non_header_with_children = coa.loc[
        ~coa["account_type"].astype("string").str.casefold().eq("header")
        & coa["account_code"].astype("string").str.strip().isin(children),
        ["account_code", "account_name", "account_type", "statement"],
    ]
    if not non_header_with_children.empty:
        add("high", "COA_NON_HEADER_WITH_CHILDREN", "A non-header COA account also has children. Confirm posting/roll-up semantics to avoid double counting.", _records(non_header_with_children, list(non_header_with_children.columns)))

    tbd_cf = coa.loc[
        coa["cf_category"].astype("string").str.strip().str.casefold().eq("tbd"),
        ["account_code", "account_name", "cf_category", "statement"],
    ]
    if not tbd_cf.empty:
        add("medium", "COA_CF_CATEGORY_TBD", "Cash-flow categories explicitly marked TBD require a policy decision or human review.", _records(tbd_cf, list(tbd_cf.columns)))

    # Prior-period codes and names are compared against the current COA without assuming a mapping.
    prior_not_in_coa = prior.loc[~prior["account_code"].astype("string").str.strip().isin(coa_codes), ["account_code", "account_name", "currency"]]
    if not prior_not_in_coa.empty:
        add("medium", "PRIOR_ACCOUNT_NOT_IN_CURRENT_COA", "Prior-period accounts are absent from the current COA; candidate mappings require evidence and approval.", _records(prior_not_in_coa.drop_duplicates(subset=["account_code", "account_name"]), ["account_code", "account_name", "currency"]))

    coa_name_by_code = coa.set_index("account_code")["account_name"].to_dict()
    current_names = tb[["account_code", "account_name"]].drop_duplicates()
    name_mismatch_rows = []
    for row in current_names.itertuples(index=False):
        code = str(row.account_code).strip()
        coa_name = coa_name_by_code.get(code)
        if coa_name is not None and pd.notna(coa_name) and str(row.account_name).strip() != str(coa_name).strip():
            name_mismatch_rows.append({"account_code": code, "tb_name": row.account_name, "coa_name": coa_name})
    if name_mismatch_rows:
        add("low", "TB_COA_NAME_MISMATCH", "Account codes present in the COA have different names in the trial balance; preserve both names for review.", name_mismatch_rows)

    # FX rates: require period_end for every non-functional currency present in the TB.
    functional_currency = str(adjustments.get("functional_currency", "USD")).strip()
    tb_currencies = set(tb["currency"].dropna().astype(str).str.strip())
    end_rate_currencies = set(
        fx.loc[fx["rate_type"].astype("string").str.casefold().eq("period_end"), "currency"]
        .dropna().astype(str).str.strip()
    )
    missing_end_rates = sorted(c for c in tb_currencies if c != functional_currency and c not in end_rate_currencies)
    if missing_end_rates:
        add("high", "FX_PERIOD_END_RATE_MISSING", "Non-functional currencies in the current TB lack a period-end FX rate. Do not silently substitute period-average or opening rates.", {"functional_currency": functional_currency, "currencies": missing_end_rates})

    # Manual adjustment controls: balance and COA references.
    adjustment_entries = adjustments.get("entries", [])
    adjustment_account_refs = []
    unbalanced_entries = []
    same_account_entries = []
    malformed_entries = []
    for entry in adjustment_entries:
        entry_id = entry.get("id", "UNKNOWN")
        lines = entry.get("lines", [])
        if not lines:
            malformed_entries.append({"entry_id": entry_id, "reason": "no_lines"})
            continue
        debit_total = sum(float(line.get("debit", 0) or 0) for line in lines)
        credit_total = sum(float(line.get("credit", 0) or 0) for line in lines)
        difference = round(debit_total - credit_total, 2)
        if difference != 0:
            unbalanced_entries.append({"entry_id": entry_id, "debit_total": debit_total, "credit_total": credit_total, "difference": difference})
        line_accounts = [str(line.get("account", "")).strip() for line in lines]
        unknown = sorted(set(line_accounts) - coa_codes)
        if unknown:
            adjustment_account_refs.append({"entry_id": entry_id, "unknown_account_codes": unknown})
        if len(line_accounts) != len(set(line_accounts)):
            same_account_entries.append({"entry_id": entry_id, "description": entry.get("description", ""), "account_codes": line_accounts})
    if unbalanced_entries:
        add("high", "ADJUSTMENT_OUT_OF_BALANCE", "One or more proposed journal entries have unequal debit and credit totals; quarantine rather than post.", unbalanced_entries)
    if adjustment_account_refs:
        add("high", "ADJUSTMENT_ACCOUNT_NOT_IN_COA", "Proposed journal entries reference accounts absent from the COA.", adjustment_account_refs)
    if same_account_entries:
        add("medium", "ADJUSTMENT_REPEATED_ACCOUNT_LINES", "A journal entry debits and credits the same account code on multiple lines. This is not automatically invalid, but requires semantic review, especially for intercompany entries.", same_account_entries)
    if malformed_entries:
        add("high", "ADJUSTMENT_MALFORMED", "One or more entries contain no journal lines.", malformed_entries)

    summary = {
        "files": {
            "trial_balance_rows": int(len(tb)),
            "chart_of_accounts_rows": int(len(coa)),
            "prior_period_rows": int(len(prior)),
            "fx_rate_rows": int(len(fx)),
            "adjustment_entries": int(len(adjustment_entries)),
        },
        "currencies_in_trial_balance": sorted(tb_currencies),
        "functional_currency": functional_currency,
        "finding_count": len(findings),
        "findings_by_severity": {
            severity: sum(1 for item in findings if item["severity"] == severity)
            for severity in ["info", "low", "medium", "high", "error"]
        },
        "findings": findings,
        "note": "Read-only diagnostic report. Findings are signals for review, not automatic accounting conclusions.",
    }
    return summary


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Persist the report as JSON without modifying source inputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
