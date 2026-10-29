import uuid
from pathlib import Path

from langgraph.types import Command

from finance_mapper.agent_graph import build_agent_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ask_nonempty(prompt: str) -> str:
    """Require a non-empty reviewer response."""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("This field cannot be empty.")


def main() -> None:
    graph = build_agent_graph()

    # A thread ID connects the paused and resumed executions.
    config = {
        "configurable": {
            "thread_id": f"coa-review-{uuid.uuid4()}",
        }
    }

    print("\nStarting the COA mapping agent...")
    print("This run will call Gemini for the unresolved accounts.")

    result = graph.invoke(
        {
            "input_dir": str(INPUT_DIR),
            "output_dir": str(OUTPUT_DIR),
            "top_k": 3,
        },
        config=config,
    )

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        raise RuntimeError(
            "The graph did not pause for human review. "
            "Inspect the graph state and execution output."
        )

    # Read the saved graph state while it is paused.
    snapshot = graph.get_state(config)
    recommendations = snapshot.values.get("recommendations", [])

    if not recommendations:
        raise RuntimeError(
            "The graph paused without any mapping recommendations."
        )

    print("\n" + "=" * 60)
    print("HUMAN REVIEW QUEUE")
    print("=" * 60)

    decisions = []

    for item in recommendations:
        print(f"\nSource account: {item['source_account_code']}")
        print(f"Source name: {item.get('source_account_name')}")
        print(f"LLM decision: {item['decision']}")
        print(
            "Suggested code:",
            item.get("recommended_account_code") or "ABSTAIN",
        )
        print(f"Reason: {item['reason']}")

        print("\nRetrieved candidates:")
        for candidate in item["candidates"]:
            print(
                f"  {candidate['account_code']} - "
                f"{candidate['account_name']} "
                f"(cosine={candidate['similarity_score']:.4f})"
            )

        allowed_codes = {
            str(candidate["account_code"])
            for candidate in item["candidates"]
        }

        while True:
            action = input(
                "\nAction [approve/reject/quarantine]: "
            ).strip().lower()

            if action in {"approve", "reject", "quarantine"}:
                break

            print("Enter approve, reject, or quarantine.")

        target_code = None

        if action == "approve":
            while True:
                target_code = input(
                    "Candidate account code to approve: "
                ).strip()

                if target_code in allowed_codes:
                    break

                print(
                    "Invalid target. Choose a code from this account's "
                    "retrieved candidate list."
                )

        reviewer = ask_nonempty("Reviewer name: ")
        notes = ask_nonempty("Review notes: ")

        decisions.append({
            "source_account_code": item["source_account_code"],
            "action": action,
            "target_account_code": target_code,
            "reviewer": reviewer,
            "notes": notes,
        })

    print("\nResuming the graph with reviewer decisions...")

    final_state = graph.invoke(
        Command(resume={"decisions": decisions}),
        config=config,
    )

    report_path = OUTPUT_DIR / "agent_review_results.json"

    print("\nAgent workflow completed.")
    print(f"Review report: {report_path}")

    review_results = final_state.get("final_results", {}).get(
        "review_decisions", []
    )

    for decision in review_results:
        print(
            f"{decision['source_account_code']}: "
            f"{decision['review_action']}"
        )


if __name__ == "__main__":
    main()