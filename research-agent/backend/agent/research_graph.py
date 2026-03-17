import logging
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Optional
import asyncio
 
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
 
from models.inference import inference_engine
from models.model_manager import ModelSize
from tools.scraper import scraper, Source
from core.config import get_settings
 
log = logging.getLogger(__name__)
settings = get_settings() 


 
# ── State ────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    # Input
    query:       str
    model_size:  str                        # "0.6b" | "1.5b"
    session_id:  str
 
    # Planning
    sub_queries: list[str]
 
    # Scraping
    raw_sources: list[dict]                 # serialised Source objects
 
    # Analysis
    source_summaries: list[dict]            # {url, title, summary, facts}
 
    # Output
    final_report:    Optional[str]
    key_findings:    list[str]
    sources_used:    list[dict]
    status:          str                    # planning|scraping|analysing|synthesizing|done|error
    error:           Optional[str]
 
    # Streaming messages
    messages: Annotated[list, add_messages]
 
 
# ── Node: Planner ─────────────────────────────────────────────────────────────
def node_planner(state: ResearchState) -> dict:
    log.info(f"[Planner] Decomposing: {state['query']}")
    model_size = ModelSize(state["model_size"])
 
    try:
        sub_queries = inference_engine.decompose_query(state["query"], model_size)
        # Always include the original query
        if state["query"] not in sub_queries:
            sub_queries.insert(0, state["query"])
        sub_queries = sub_queries[:5]   # cap at 5
    except Exception as e:
        log.error(f"Planner error: {e}")
        sub_queries = [state["query"]]  # fallback to original
 
    log.info(f"[Planner] Generated {len(sub_queries)} sub-queries")
    return {
        "sub_queries": sub_queries,
        "status": "scraping",
        "messages": [{"role": "system", "content": f"Planning complete. {len(sub_queries)} sub-queries generated."}],
    }
 
 
# ── Node: Scraper ─────────────────────────────────────────────────────────────
def node_scraper(state: ResearchState) -> dict:
    log.info(f"[Scraper] Fetching {len(state['sub_queries'])} queries")
 
    try:
        sources: list[Source] = asyncio.run(
            scraper.search_batch(state["sub_queries"])
        )
        raw = [
            {
                "url":            s.url,
                "title":          s.title,
                "content":        s.truncated_content(2500),
                "score":          s.score,
                "published_date": s.published_date,
            }
            for s in sources[:settings.TAVILY_MAX_RESULTS]
        ]
    except Exception as e:
        log.error(f"Scraper error: {e}")
        raw = []
 
    log.info(f"[Scraper] Retrieved {len(raw)} sources")
    return {
        "raw_sources": raw,
        "status": "analysing",
        "messages": [{"role": "system", "content": f"Scraped {len(raw)} sources."}],
    }
 
 
# ── Node: Analyser ────────────────────────────────────────────────────────────
def node_analyser(state: ResearchState) -> dict:
    log.info(f"[Analyser] Analysing {len(state['raw_sources'])} sources")
    model_size = ModelSize(state["model_size"])
    summaries = []
 
    for src in state["raw_sources"]:
        if not src.get("content"):
            continue
        try:
            summary = inference_engine.summarize(src["content"], model_size)
            facts   = inference_engine.extract_facts(src["content"], model_size)
            summaries.append({
                "url":     src["url"],
                "title":   src["title"],
                "summary": summary,
                "facts":   facts,
                "score":   src.get("score", 0),
            })
        except Exception as e:
            log.warning(f"Analysis failed for {src['url']}: {e}")
            continue
 
    log.info(f"[Analyser] Analysed {len(summaries)} sources")
    return {
        "source_summaries": summaries,
        "status": "synthesizing",
        "messages": [{"role": "system", "content": f"Analysis complete. {len(summaries)} sources processed."}],
    }
 
 
# ── Node: Synthesizer ─────────────────────────────────────────────────────────
def node_synthesizer(state: ResearchState) -> dict:
    log.info("[Synthesizer] Generating final report")
    model_size = ModelSize(state["model_size"])
 
    # Build combined findings text
    findings_parts = [f"Research Query: {state['query']}\n"]
    all_facts = []
 
    for i, src in enumerate(state["source_summaries"], 1):
        findings_parts.append(
            f"\nSource {i}: {src['title']}\n"
            f"Summary: {src['summary']}\n"
        )
        all_facts.extend(src.get("facts", []))
 
    combined_findings = "\n".join(findings_parts)
 
    try:
        report = inference_engine.synthesize_report(combined_findings, model_size)
    except Exception as e:
        log.error(f"Synthesis error: {e}")
        report = combined_findings
 
    # Deduplicate facts
    seen = set()
    unique_facts = []
    for f in all_facts:
        key = f.lower().strip()
        if key not in seen and len(key) > 20:
            seen.add(key)
            unique_facts.append(f)
 
    sources_used = [
        {"url": s["url"], "title": s["title"], "score": s.get("score", 0)}
        for s in state["source_summaries"]
    ]
 
    log.info("[Synthesizer] Report generated")
    return {
        "final_report": report,
        "key_findings": unique_facts[:10],
        "sources_used": sources_used,
        "status": "done",
        "messages": [{"role": "system", "content": "Research complete."}],
    }
 
 
# ── Error Handler ─────────────────────────────────────────────────────────────
def node_error_handler(state: ResearchState) -> dict:
    return {"status": "error", "error": "An unexpected error occurred."}
 
 
# ── Build Graph ───────────────────────────────────────────────────────────────
def build_research_graph():
    graph = StateGraph(ResearchState)
 
    graph.add_node("planner",     node_planner)
    graph.add_node("scraper",     node_scraper)
    graph.add_node("analyser",    node_analyser)
    graph.add_node("synthesizer", node_synthesizer)
 
    graph.set_entry_point("planner")
    graph.add_edge("planner",     "scraper")
    graph.add_edge("scraper",     "analyser")
    graph.add_edge("analyser",    "synthesizer")
    graph.add_edge("synthesizer", END)
 
    return graph.compile()
 
 
research_graph = build_research_graph()