from finance_mapper.schemas import MappingDecision

decision = MappingDecision(
    source_account_code="6310",
    source_account_name="Travel and Entertainment",
    target_account_code="6310",
    target_account_name="Travel and Entertainment",
    status="exact_match",
    method="exact_code",
    confidence_score=1.0,
    reason="The account code matches a valid COA posting account.",
    evidence=["Exact account-code match in the current COA."],
    requires_human_review=False,
)

print(decision.model_dump_json(indent=2))