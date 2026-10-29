
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from .agent_graph import build_agent_graph
from .schemas import HumanReviewDecision, HumanReviewSubmission

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "inputs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Keep one compiled graph alive so its in-memory checkpoints can be
# found by subsequent review requests in this same server process.
graph = build_agent_graph()

app = FastAPI(
    title="Finance Mapping Agent API",
    description=(
        "Trial Balance to Chart of Accounts mapping with hybrid retrieval, "
        "LLM recommendations, and human review."
    ),
    version="0.1.0",
)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1)
    decisions: list[HumanReviewDecision] = Field(min_length=1)


@app.get("/")
def root() -> dict:
    return {
        "name": "Finance Mapping Agent API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "finance-mapping-agent"}


@app.post("/api/run")
def run_agent() -> dict[str, Any]:
    """Start a mapping run and return its pending review queue."""

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        graph.invoke(
            {
                "input_dir": str(INPUT_DIR),
                "output_dir": str(OUTPUT_DIR),
                "top_k": 3,
            },
            config=config,
        )

        snapshot = graph.get_state(config)
        recommendations = snapshot.values.get("recommendations", [])

        if not snapshot.next:
            raise HTTPException(
                status_code=500,
                detail="The workflow did not pause for human review.",
            )

        return {
            "status": "awaiting_review",
            "thread_id": thread_id,
            "review_items": recommendations,
            "message": (
                "Review the recommendations and submit one decision per "
                "source account to POST /api/review."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent run failed")
        raise HTTPException(
            status_code=500,
            detail=f"Agent run failed ({type(exc).__name__}). "
                   "Check the server console for details.",
        ) from exc


@app.post("/api/review")
def submit_review(request: ReviewRequest) -> dict[str, Any]:
    """Resume a paused graph with reviewer decisions."""

    config = {
        "configurable": {"thread_id": request.thread_id}
    }

    try:
        snapshot = graph.get_state(config)

        if not snapshot.values or not snapshot.next:
            raise HTTPException(
                status_code=404,
                detail="No pending review was found for this thread ID.",
            )

        submission = HumanReviewSubmission(
            decisions=request.decisions
        )

        final_state = graph.invoke(
            Command(resume=submission.model_dump(mode="json")),
            config=config,
        )

        final_report = final_state.get("final_results")

        if not final_report:
            final_report = graph.get_state(config).values.get(
                "final_results"
            )

        if not final_report:
            raise HTTPException(
                status_code=500,
                detail="The workflow resumed but did not produce a final report.",
            )

        return {
            "status": "completed",
            "thread_id": request.thread_id,
            "review_decisions": final_report.get("review_decisions", []),
            "report_path": str(
                OUTPUT_DIR / "agent_review_results.json"
            ),
        }

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Review submission failed")
        raise HTTPException(
            status_code=500,
            detail=f"Review submission failed ({type(exc).__name__}). "
                   "Check the server console for details.",
        ) from exc


@app.get("/api/latest")
def latest_report() -> dict[str, Any]:
    """Return the most recent completed agent review report."""

    report_path = OUTPUT_DIR / "agent_review_results.json"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No completed agent review report exists yet.",
        )

    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Unable to read the latest report")
        raise HTTPException(
            status_code=500,
            detail="The latest report could not be read.",
        ) from exc
