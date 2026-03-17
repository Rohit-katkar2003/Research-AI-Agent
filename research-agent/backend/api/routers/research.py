"""
api/routers/research.py — /research endpoints
"""
import time
import uuid
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json

from api.schemas import ResearchRequest, ResearchResponse, SourceItem
from agents.research_graph import research_graph
from core.config import get_settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])
settings = get_settings()


@router.post("/run", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    """
    Execute full research pipeline:
    plan → scrape → analyse → synthesize
    """
    session_id = str(uuid.uuid4())
    t0 = time.time()

    log.info(f"[{session_id}] Starting research: '{req.query}' | model={req.model_size}")

    initial_state = {
        "query":            req.query,
        "model_size":       req.model_size,
        "session_id":       session_id,
        "sub_queries":      [],
        "raw_sources":      [],
        "source_summaries": [],
        "final_report":     None,
        "key_findings":     [],
        "sources_used":     [],
        "status":           "planning",
        "error":            None,
        "messages":         [],
    }

    try:
        result = await research_graph.ainvoke(initial_state)
    except Exception as e:
        log.error(f"[{session_id}] Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    duration = round(time.time() - t0, 2)
    log.info(f"[{session_id}] Complete in {duration}s")

    return ResearchResponse(
        session_id   = session_id,
        query        = req.query,
        model_used   = req.model_size,
        status       = result["status"],
        final_report = result.get("final_report"),
        key_findings = result.get("key_findings", []),
        sources_used = [SourceItem(**s) for s in result.get("sources_used", [])],
        sub_queries  = result.get("sub_queries", []),
        duration_sec = duration,
    )


@router.post("/stream")
async def stream_research(req: ResearchRequest):
    """
    SSE streaming endpoint — emits progress events as each node completes.
    Frontend can subscribe and update UI in real-time.
    """
    session_id = str(uuid.uuid4())

    async def event_generator():
        initial_state = {
            "query":            req.query,
            "model_size":       req.model_size,
            "session_id":       session_id,
            "sub_queries":      [],
            "raw_sources":      [],
            "source_summaries": [],
            "final_report":     None,
            "key_findings":     [],
            "sources_used":     [],
            "status":           "planning",
            "error":            None,
            "messages":         [],
        }

        node_labels = {
            "planner":     "🧠 Planning research strategy...",
            "scraper":     "🔍 Scraping web sources...",
            "analyser":    "⚙️  Analysing sources with AI...",
            "synthesizer": "📝 Synthesizing final report...",
        }

        try:
            async for event in research_graph.astream_events(initial_state, version="v2"):
                name = event.get("name", "")
                kind = event.get("event", "")

                if kind == "on_chain_start" and name in node_labels:
                    data = json.dumps({"type": "progress", "node": name, "message": node_labels[name]})
                    yield f"data: {data}\n\n"

                elif kind == "on_chain_end" and name == "synthesizer":
                    output = event.get("data", {}).get("output", {})
                    data = json.dumps({
                        "type":         "complete",
                        "session_id":   session_id,
                        "final_report": output.get("final_report"),
                        "key_findings": output.get("key_findings", []),
                        "sources_used": output.get("sources_used", []),
                        "sub_queries":  output.get("sub_queries", []),
                    })
                    yield f"data: {data}\n\n"

        except Exception as e:
            data = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")