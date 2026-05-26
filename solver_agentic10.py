"""
============================================================
SolverSearch Backend v2.2 (FastAPI + LangGraph)
============================================================
Discovers THREE kinds of solutions in parallel:

  1. EXPERTS    — LinkedIn profiles (people who can solve the problem)
  2. TOOLS      — Software / SaaS / apps (ProductHunt, G2, Capterra, etc.)
  3. BUSINESSES — Agencies, consultancies, firms (Clutch, GoodFirms, etc.)

CHANGES in v2.2:
  • Refinement is now "first search, again, with better input."
    The frontend builds an updated `description` from the full chat history
    and POSTs it as a regular request. No more `seen_urls`, `previous_queries`,
    or `is_continuation` plumbing — the entire diagnosis + classification +
    search + synthesis pipeline runs from scratch on every call.
  • HOLISTIC COMPARISON preserved: /compare_candidates and /compare_solutions
    accept an optional `other_resources` payload (tools + agencies discovered
    in the same search) so the strategic recommendation can suggest a full
    SOLUTION STACK — expert + tool + agency — instead of just experts.
  • Error resilience preserved: every branch is wrapped in try/except, every
    SerpAPI call has bounded failure, and the stream always emits a `done`
    event even if the graph crashes mid-flight.

Pipeline for /find_solutions_agentic:

    classify_needs ──┬─► experts_branch
                     ├─► tools_branch
                     └─► businesses_branch
                              │
                              ▼
                     synthesize_stack  → "Solution Stack" recommendation

Run:
    python solver_agentic_v2.py
"""

import os
import uuid
import asyncio
import httpx
import re
import logging
import json
import operator
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, List, Optional, Tuple, Literal, TypedDict, Annotated

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI
import uvicorn
from weasyprint import HTML, CSS

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from fastapi.staticfiles import StaticFiles


from fastapi.templating import Jinja2Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app = FastAPI(title="SolverSearch Engine v2.2")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ------------------------------ Configuration ------------------------------
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
SERP_API_KEY        = os.getenv("SERP_API_KEY", "")
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY", "")


if not all([OPENAI_API_KEY, SERP_API_KEY, SCRAPINGDOG_API_KEY]):
    logging.warning("One or more API keys are missing. Set OPENAI_API_KEY, SERP_API_KEY, SCRAPINGDOG_API_KEY.")

# Per-branch search budget.
NUM_QUERIES_EXPERTS    = 15
NUM_QUERIES_TOOLS      = 8
NUM_QUERIES_BUSINESSES = 8
NUM_RESULTS_PER_QUERY  = 90

client = OpenAI(api_key=OPENAI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solversearch")

SERP_CONCURRENCY    = int(os.getenv("SERP_CONCURRENCY", "3"))
SCRAPE_CONCURRENCY  = int(os.getenv("SCRAPE_CONCURRENCY", "1"))
_serp_sem    = asyncio.Semaphore(SERP_CONCURRENCY)
_scrape_sem  = asyncio.Semaphore(SCRAPE_CONCURRENCY)

# ============================================================
# Schemas
# ============================================================

class PDFRequest(BaseModel):
    html_content: str

class JobRequest(BaseModel):
    """
    Simple, single-purpose request body. Every search — first or refinement —
    sends the same shape. The frontend is responsible for assembling a complete,
    self-contained `description` that captures the full problem context.
    """
    description: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class CompareRequest(BaseModel):
    description: str
    profiles: List[Dict[str, Any]]
    # Pass top tools / agencies from the same search so the strategic
    # recommendation can suggest a SOLUTION STACK, not just experts.
    other_resources: Optional[Dict[str, List[Dict[str, Any]]]] = None

class CompareSolutionsRequest(BaseModel):
    """Generic compare across mixed resource types (experts, tools, businesses)."""
    description: str
    items: List[Dict[str, Any]]
    other_resources: Optional[Dict[str, List[Dict[str, Any]]]] = None

class ChatTurnResponse(BaseModel):
    status: Literal["chatting", "complete"] = Field(
        description="'chatting' if more info needed, 'complete' if diagnosed."
    )
    reply: Optional[str] = Field(default=None)
    options: Optional[List[str]] = Field(default=[])
    final_description: Optional[str] = Field(default=None)

class SolutionClassification(BaseModel):
    """Decides which solution surfaces to search."""
    needs_experts:   bool = Field(description="True if the problem needs a human expert / advisor / consultant.")
    needs_tools:     bool = Field(description="True if a software product / SaaS / app would help solve this.")
    needs_businesses:bool = Field(description="True if an agency / firm / consultancy is a fit.")
    expert_focus:    str  = Field(default="", description="One-line description of the kind of expert to find.")
    tool_focus:      str  = Field(default="", description="One-line description of the kind of tool category.")
    business_focus:  str  = Field(default="", description="One-line description of the kind of agency / firm.")
    reasoning:       str  = Field(default="", description="Why these branches were selected.")

# ============================================================
# Query generation helpers
# ============================================================

async def generate_search_queries(description: str, feedback: str = None) -> list[str]:
    """LinkedIn-targeted queries for the EXPERTS branch."""
    prompt = f"""
    Generate {NUM_QUERIES_EXPERTS} highly effective Google search queries to find LinkedIn profiles
    (using site:linkedin.com/in/) of EXPERTS, ADVISORS, or CONSULTANTS who can solve the following problem:

    "{description}"

    Focus on keywords related to the specific skills, domain (e.g., B2B SaaS), and problem-solving
    experience. Include terms like "Advisor", "Consultant", "Freelance", or specific senior titles if applicable.
    """
    if feedback:
        prompt += f"\n\nWARNING - Previous search failed with this feedback: {feedback}\n"
        prompt += "Please ADJUST your search strategy. Use different keywords, synonyms, or look for different job titles."
    prompt += "\nReturn only the queries, one per line."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7 if feedback else 0.4,
    )
    text = resp.choices[0].message.content.strip()
    return [q.strip() for q in text.splitlines() if q.strip()]


PROFILE_RE = re.compile(r"https?://(?:[\w-]+\.)*linkedin\.com/(?:in|pub)/[^/?#]+/?", re.IGNORECASE)

def _canonicalize(url: str) -> str | None:
    m = PROFILE_RE.match(url)
    if not m: return None
    u = urlparse(m.group(0))
    path = u.path if u.path.endswith("/") else u.path + "/"
    return urlunparse((u.scheme, u.netloc.lower(), path, "", "", ""))


async def fetch_serpapi_results(query: str, profile_filter: bool = True) -> list:
    """
    Returns SerpAPI organic results.
    - profile_filter=True returns canonicalized LinkedIn profile URLs (strings).
    - profile_filter=False returns raw organic_results dicts (title/link/snippet).

    HARDENED: every network/parse failure becomes an empty list, never a raised
    exception. A single bad SerpAPI call must not take down the whole branch.
    """
    serp_url = "https://serpapi.com/search.json"
    params = {
        "engine": "google", "q": query, "num": NUM_RESULTS_PER_QUERY,
        "api_key": SERP_API_KEY, "hl": "en", "gl": "us", "google_domain": "google.com",
    }
    try:
        async with _serp_sem:
            async with httpx.AsyncClient(timeout=130.0) as ac:
                resp = await ac.get(serp_url, params=params)
    except httpx.TimeoutException as e:
        logger.warning(f"SerpAPI timeout for query={query!r}: {e}")
        return []
    except Exception as e:
        logger.warning(f"SerpAPI network error for query={query!r}: {e}")
        return []

    if resp.status_code != 200:
        logger.warning(f"SerpAPI non-200 ({resp.status_code}) for query={query!r}: {resp.text[:200]}")
        return []
    try:
        data = resp.json()
    except Exception as e:
        logger.warning(f"SerpAPI bad JSON for query={query!r}: {e}")
        return []
    organic = data.get("organic_results", [])

    if profile_filter:
        urls = set()
        for item in organic:
            if link := item.get("link"):
                if canon := _canonicalize(link):
                    urls.add(canon)
        return list(urls)
    return [
        {
            "title":   item.get("title", ""),
            "link":    item.get("link", ""),
            "snippet": item.get("snippet", ""),
            "source":  item.get("source", ""),
        }
        for item in organic
        if item.get("link")
    ]


async def scrape_profile(profile_url: str) -> Dict[str, Any]:
    """ScrapingDog's LinkedIn-specific endpoint. Used by experts branch only."""
    try:
        profile_id = urlparse(profile_url).path.strip("/").split('/')[-1]
    except Exception:
        profile_id = ''

    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "type": "profile",
        "linkId": profile_id,
        "premium": "true"
    }
    api_url = "https://api.scrapingdog.com/linkedin/"
    result: Dict[str, Any] = {"_profile_url": profile_url}

    try:
        async with _scrape_sem:
            async with httpx.AsyncClient(timeout=60.0) as ac:
                resp = await ac.get(api_url, params=params)
        if resp.status_code == 200:
            try:
                data = resp.json()
                logger.info(f"Successfully scraped {profile_id}")
                if isinstance(data, list) and len(data) > 0:
                    result.update(data[0])
                elif isinstance(data, dict):
                    result.update(data)
            except Exception as e:
                logger.error(f"Failed to parse JSON for {profile_id}: {e}")
        elif resp.status_code == 202:
            logger.warning(f"Profile {profile_id} queued by Scrapingdog (202).")
        else:
            logger.warning(f"Scraping failed for {profile_id} with status {resp.status_code}.")
    except httpx.ReadTimeout:
        logger.error(f"Scraping API timed out for {profile_url}")
    except Exception as e:
        logger.error(f"Unexpected error scraping {profile_url}: {e}")
    return result


async def scrape_generic_url(url: str, max_chars: int = 4000) -> str:
    """
    ScrapingDog's general-purpose scraper. Used by tools/businesses branches when we
    want richer context than the SerpAPI snippet. Returns plain text (truncated).
    """
    params = {
        "api_key": SCRAPINGDOG_API_KEY,
        "url": url,
        "dynamic": "false",
    }
    api_url = "https://api.scrapingdog.com/scrape"
    try:
        async with _scrape_sem:
            async with httpx.AsyncClient(timeout=45.0) as ac:
                resp = await ac.get(api_url, params=params)
        if resp.status_code == 200:
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:max_chars]
    except Exception as e:
        logger.warning(f"Generic scrape failed for {url}: {e}")
    return ""


def extract_popularity(profile_json: Dict[str, Any]) -> int:
    candidates = []
    for key in ("followers", "connections", "followers_count"):
        v = profile_json.get(key)
        try:
            if isinstance(v, str):
                v = int(re.sub(r"[^0-9]", "", v) or 0)
            elif isinstance(v, (int, float)):
                v = int(v)
            if v and v > 0:
                candidates.append(v)
        except Exception:
            pass
    return max(candidates) if candidates else 0


async def score_profile_with_llm(description: str, profile_json: Dict[str, Any]) -> Tuple[int, str, str]:
    prompt = (
        "Score how well this expert's LinkedIn profile proves they can SOLVE the problem described below. "
        "Rate strictly on a scale from 1 to 1000 (e.g., 850, 920). Do NOT use a 1-10 scale.\n"
        "Also, generate a 3-5 word 'dynamic_title' summarizing their specific expertise related to the problem "
        "(e.g., 'B2B SaaS Growth Consultant', 'Data Pipeline Architect').\n"
        "Output ONLY JSON: {'score': integer, 'reason': 'short explanation', 'dynamic_title': 'string'}.\n\n"
        f"Problem/Expert Required: {description}\n\nProfile JSON: {json.dumps(profile_json, default=str)[:3000]}\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        return (
            min(max(int(parsed.get('score', 1)), 1), 1000),
            parsed.get('reason', ''),
            parsed.get('dynamic_title', 'Specialized Expert')
        )
    except Exception:
        return 0, "Failed to parse LLM score.", "Expert / Consultant"


async def _process_single_profile(link: str, description: str) -> dict | None:
    prof_json = await scrape_profile(link)
    if not prof_json:
        return None
    score, reason, dynamic_title = await score_profile_with_llm(description, prof_json)
    return {
        "resource_type": "expert",
        "profile_url": prof_json.get("_profile_url"),
        "popularity": extract_popularity(prof_json),
        "llm_score": score,
        "llm_reason": reason,
        "dynamic_title": dynamic_title,
        "profile_json": prof_json,
    }


# ============================================================
# Tool discovery helpers
# ============================================================

TOOL_SOURCES = [
    "producthunt.com",
    "g2.com",
    "capterra.com",
    "alternativeto.net",
    "getapp.com",
    "softwareadvice.com",
]

async def generate_tool_queries(description: str, focus: str) -> List[str]:
    """Targeted queries against tool directories + generic 'best X tool' queries."""
    prompt = f"""
    Generate {NUM_QUERIES_TOOLS} Google search queries to discover SOFTWARE TOOLS, SaaS PRODUCTS, or APPS
    that would help solve the following problem.

    Problem: "{description}"
    Tool category hint: "{focus}"

    Mix these query patterns:
      - site:producthunt.com {{category}}
      - site:g2.com {{category}}
      - site:capterra.com {{category}}
      - site:alternativeto.net {{specific tool name}}
      - "best {{category}} tool for {{use_case}}" 2026
      - "top {{category}} software" 2026

    Return ONLY the queries, one per line. No numbering, no commentary.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return [q.strip() for q in resp.choices[0].message.content.strip().splitlines() if q.strip()]


async def score_tool_with_llm(description: str, item: Dict[str, Any], scraped_text: str = "") -> Tuple[int, str, str, str]:
    """Returns: (score 1-1000, reason, dynamic_title, extracted_tool_name)"""
    prompt = (
        "You are evaluating whether a web search result describes a SOFTWARE TOOL useful for solving the problem.\n"
        "Score 1-1000 how well this tool fits the problem.\n"
        "If the result is a listicle or directory page, score the OVERALL relevance of the page.\n"
        "Extract the primary tool name. If it's a listicle, return the most prominent tool name as 'tool_name'.\n"
        "Output ONLY JSON: "
        "{'score': int, 'reason': 'short why', 'dynamic_title': '3-5 words on what it does', 'tool_name': 'string'}.\n\n"
        f"Problem: {description}\n\n"
        f"Search result:\n  Title: {item.get('title')}\n  URL: {item.get('link')}\n  Snippet: {item.get('snippet')}\n\n"
        f"Page text (truncated):\n{scraped_text[:2000]}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        return (
            min(max(int(parsed.get('score', 1)), 1), 1000),
            parsed.get('reason', ''),
            parsed.get('dynamic_title', 'Software Tool'),
            parsed.get('tool_name', item.get('title', 'Unknown Tool')),
        )
    except Exception:
        return 0, "Failed to parse LLM score.", "Software Tool", item.get('title', 'Unknown')


async def _process_single_tool(item: Dict[str, Any], description: str) -> dict | None:
    url = item.get("link", "")
    domain = urlparse(url).netloc.lower()

    scraped = ""
    if any(src in domain for src in TOOL_SOURCES):
        scraped = await scrape_generic_url(url)

    score, reason, dynamic_title, tool_name = await score_tool_with_llm(description, item, scraped)
    if score < 200:
        return None
    return {
        "resource_type": "tool",
        "profile_url": url,
        "tool_name": tool_name,
        "llm_score": score,
        "llm_reason": reason,
        "dynamic_title": dynamic_title,
        "source_domain": domain,
        "snippet": item.get("snippet", ""),
        "page_excerpt": scraped[:600] if scraped else "",
    }


# ============================================================
# Business / agency discovery helpers
# ============================================================

BUSINESS_SOURCES = [
    "clutch.co",
    "goodfirms.co",
    "designrush.com",
    "linkedin.com/company",
    "upwork.com/agencies",
    "crunchbase.com/organization",
]

async def generate_business_queries(description: str, focus: str) -> List[str]:
    prompt = f"""
    Generate {NUM_QUERIES_BUSINESSES} Google search queries to discover AGENCIES, CONSULTANCIES, or FIRMS
    that solve the following problem for clients.

    Problem: "{description}"
    Agency focus hint: "{focus}"

    Mix these query patterns:
      - site:clutch.co {{service}} {{industry}}
      - site:goodfirms.co {{service}}
      - site:designrush.com {{service}}
      - "top {{service}} agencies for {{industry}}" 2026
      - "best {{service}} consulting firms" 2026
      - site:linkedin.com/company {{service}}

    Return ONLY the queries, one per line. No numbering, no commentary.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return [q.strip() for q in resp.choices[0].message.content.strip().splitlines() if q.strip()]


async def score_business_with_llm(description: str, item: Dict[str, Any], scraped_text: str = "") -> Tuple[int, str, str, str]:
    prompt = (
        "You are evaluating whether a web search result describes an AGENCY, CONSULTANCY, or FIRM useful for "
        "solving the problem.\n"
        "Score 1-1000 how well this firm fits the problem.\n"
        "Extract the primary firm name as 'firm_name'.\n"
        "Output ONLY JSON: "
        "{'score': int, 'reason': 'short why', 'dynamic_title': '3-5 words on their specialization', "
        "'firm_name': 'string'}.\n\n"
        f"Problem: {description}\n\n"
        f"Search result:\n  Title: {item.get('title')}\n  URL: {item.get('link')}\n  Snippet: {item.get('snippet')}\n\n"
        f"Page text (truncated):\n{scraped_text[:2000]}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, response_format={"type": "json_object"}
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        return (
            min(max(int(parsed.get('score', 1)), 1), 1000),
            parsed.get('reason', ''),
            parsed.get('dynamic_title', 'Consulting Firm'),
            parsed.get('firm_name', item.get('title', 'Unknown Firm')),
        )
    except Exception:
        return 0, "Failed to parse LLM score.", "Consulting Firm", item.get('title', 'Unknown')


async def _process_single_business(item: Dict[str, Any], description: str) -> dict | None:
    url = item.get("link", "")
    domain = urlparse(url).netloc.lower()

    scraped = ""
    if any(src in domain for src in BUSINESS_SOURCES):
        scraped = await scrape_generic_url(url)

    score, reason, dynamic_title, firm_name = await score_business_with_llm(description, item, scraped)
    if score < 200:
        return None
    return {
        "resource_type": "business",
        "profile_url": url,
        "firm_name": firm_name,
        "llm_score": score,
        "llm_reason": reason,
        "dynamic_title": dynamic_title,
        "source_domain": domain,
        "snippet": item.get("snippet", ""),
        "page_excerpt": scraped[:600] if scraped else "",
    }


# ============================================================
# Shared helpers
# ============================================================

async def generate_search_summary(description: str, queries: List[str], top_profiles: List[Dict]) -> str:
    top_candidates = "\n".join(
        [f"- {p['profile_json'].get('full_name', 'Unknown')}: {p.get('llm_reason')}"
         for p in top_profiles[:3] if p.get('llm_reason')]
    )
    prompt = f"""
    You are an AI Search Coordinator. Write a brief, professional 2-3 sentence executive summary of the
    search operation you just completed.

    Diagnosed Problem: {description}
    Search Queries Used: {', '.join(queries)}
    Top Candidates Evaluated:
    {top_candidates}

    Explain what kind of experts were found. Cohesive paragraph, no bullets.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        return "Successfully executed agentic search."


def extract_name_for_llm(profile_wrapper: dict) -> str:
    prof_data = profile_wrapper.get('profile_json', {})
    for key in ['fullName', 'full_name', 'name', 'personName']:
        if prof_data.get(key):
            return str(prof_data.get(key))
    first = prof_data.get('firstName') or prof_data.get('first_name') or ''
    last  = prof_data.get('lastName')  or prof_data.get('last_name')  or ''
    if first or last:
        return f"{first} {last}".strip()
    url = profile_wrapper.get('profile_url') or prof_data.get('_profile_url') or ""
    match = re.search(r"in/([^/?#]+)", str(url))
    if match:
        return match.group(1).replace('-', ' ').title()
    return "Unknown Profile"


# ============================================================
# LangGraph #1: Chat Diagnostician
# ============================================================

class ChatGraphState(TypedDict):
    messages: List[ChatMessage]
    result: Optional[Dict[str, Any]]


CHAT_SYSTEM_PROMPT = """
You are the SolverSearch AI, an expert diagnostician and advisor for startup founders and entrepreneurs.
Users will come to you with vague business, product, growth, or operational problems
(e.g., "Users keep dropping off after onboarding").

Your goal is to ask clarifying questions to diagnose the root cause so the system can identify
the right COMBINATION of solutions — which could be an EXPERT (consultant/advisor), a TOOL (software/SaaS),
an AGENCY/FIRM, or a mix.

CRITICAL — handling refinements:
The user may already have run a search and is now providing MORE context or new constraints to refine it.
Review the ENTIRE conversation history (every prior question, answer, and refinement) and synthesize a
SINGLE COMBINED diagnosis that reflects everything they've said cumulatively. Do not treat the latest
message in isolation. The final_description you produce when status="complete" should fully capture
the current state of the problem, integrating every clarification and refinement so far — as if you
were writing the diagnosis fresh from a single rich prompt.

Review the conversation history.
- If you lack context, ask 1 or 2 concise clarifying questions. Focus on identifying:
  1. The domain/type of product (e.g., B2B SaaS, mobile app).
  2. Their current state/data maturity (e.g., "Do you track analytics?", "Have you run interviews?").
  3. Specific symptoms (e.g., "Do they drop off instantly or after a few days?").
  4. Budget / timeline constraints when relevant — this changes whether tools, experts, or agencies fit.

- When you ask a clarifying question, you MUST also provide 2 to 4 highly logical, clickable quick-reply
  options that directly answer the question. Keep them punchy (1-4 words).
  Do NOT include options like "Not sure", "Other", or "None".

OUTPUT RULES — read carefully:
- If you still need more info → set status="chatting", fill `reply` and `options`.
  Leave `final_description` empty/null. NEVER mark complete just to move on.

- If you have enough signal → set status="complete" AND you MUST populate `final_description`
  with a full 2-4 sentence paragraph describing:
    (a) the user's specific problem (domain, symptoms, current state, AND any refinements they've
        added in later turns), and
    (b) the kind of solution space — experts, tools, agencies, or a mix — that would help.
  Example final_description: "User runs a B2B SaaS for HR teams and sees ~60% drop-off between
  signup and first key action. They have basic analytics but no session replay. They've since
  clarified they have a $20k budget and want to move in under 6 weeks. They need a growth-product
  expert who has fixed activation funnels, an event-analytics tool to see where users stall, and
  prefer to avoid full-service agencies given the budget."

  An empty or missing `final_description` when status="complete" is a hard error.
  If you cannot write one, stay in status="chatting" and ask another question instead.
"""


async def _synthesize_final_description(messages: List["ChatMessage"]) -> str:
    """
    Last-resort: if the diagnostician marks the chat complete but forgets to fill
    final_description, synthesize one from the conversation history.
    """
    convo = "\n".join(f"{m.role}: {m.content}" for m in messages)
    prompt = f"""
    Based on this conversation between a user and a diagnostician, produce a single 2-4 sentence
    paragraph describing the user's PROBLEM and the kind of SOLUTION SPACE (experts, tools, agencies)
    that could help. If the user has refined or added context over multiple turns, integrate all of
    it into one combined diagnosis — don't treat the latest message in isolation.
    Output the paragraph only — no preamble, no quotes.

    Conversation:
    {convo}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        out = resp.choices[0].message.content.strip()
        if out:
            return out
    except Exception as e:
        logger.warning(f"final_description synth call failed: {e}")
    user_text = " ".join(m.content for m in messages if m.role == "user").strip()
    return user_text[:1500] or "Help me find experts, tools, or agencies to solve a business problem."


async def chat_diagnose_node(state: ChatGraphState) -> Dict[str, Any]:
    api_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for msg in state["messages"]:
        api_messages.append({"role": msg.role, "content": msg.content})

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=api_messages,
        temperature=0.6,
        response_format=ChatTurnResponse
    )
    llm_response = completion.choices[0].message.parsed

    logger.info(
        f"[Chat] status={llm_response.status}, "
        f"has_reply={bool(llm_response.reply)}, "
        f"options_count={len(llm_response.options or [])}, "
        f"has_final_desc={bool((llm_response.final_description or '').strip())}"
    )

    if llm_response.status == "chatting":
        result = {
            "status": "chatting",
            "reply": llm_response.reply or "Could you tell me a bit more about the problem?",
            "options": llm_response.options or [],
        }
    else:
        final_desc = (llm_response.final_description or "").strip()
        if not final_desc:
            logger.warning(
                "[Chat] LLM returned status=complete with empty final_description. "
                "Synthesizing one from chat history so downstream search can proceed."
            )
            final_desc = await _synthesize_final_description(state["messages"])
            logger.info(f"[Chat] Synthesized final_description: {final_desc[:120]}...")
        result = {
            "status": "complete",
            "final_description": final_desc,
        }
    return {"result": result}


_chat_g = StateGraph(ChatGraphState)
_chat_g.add_node("diagnose", chat_diagnose_node)
_chat_g.add_edge(START, "diagnose")
_chat_g.add_edge("diagnose", END)
chat_graph = _chat_g.compile()


# ============================================================
# LangGraph #2 (LEGACY): Experts-only flow (kept for /find_profiles_agentic)
# ============================================================

class FindProfilesState(TypedDict):
    description: str
    pass_num: int
    feedback: Optional[str]
    pass_queries: List[str]
    all_queries: Annotated[List[str], operator.add]
    pass_links: List[str]
    seen_urls: List[str]
    final_profiles: Annotated[List[dict], operator.add]
    best_score: int


async def generate_queries_node(state: FindProfilesState) -> Dict[str, Any]:
    writer = get_stream_writer()
    if state["pass_num"] == 1:
        writer({"type": "status", "message": "Generating search strategy..."})
        queries = await generate_search_queries(state["description"])
        writer({"type": "status", "message": f"Executing {len(queries)} queries..."})
    else:
        writer({"type": "status", "message": f"Pass 1 top score was {state['best_score']}/1000. Self-correcting..."})
        queries = await generate_search_queries(state["description"], feedback=state["feedback"])
    writer({"type": "queries", "data": queries})
    return {"pass_queries": queries, "all_queries": queries}


async def serp_search_node(state: FindProfilesState) -> Dict[str, Any]:
    writer = get_stream_writer()
    queries = state["pass_queries"]
    results_nested = await asyncio.gather(*(fetch_serpapi_results(q) for q in queries))
    all_links = sorted({url for sub in results_nested for url in sub})
    writer({"type": "found_total", "total": len(all_links)})
    if state["pass_num"] == 1:
        writer({"type": "status", "message": f"Found {len(all_links)} candidates. Scraping and evaluating..."})
    return {"pass_links": all_links}


async def process_profiles_node(state: FindProfilesState) -> Dict[str, Any]:
    writer = get_stream_writer()
    seen = set(state.get("seen_urls") or [])
    best = state.get("best_score", 0)
    new_profiles: List[dict] = []

    tasks = [_process_single_profile(link, state["description"]) for link in state["pass_links"]]
    for coro in asyncio.as_completed(tasks):
        profile = await coro
        if profile and profile["profile_url"] not in seen:
            seen.add(profile["profile_url"])
            new_profiles.append(profile)
            if state["pass_num"] == 1:
                best = max(best, profile.get("llm_score", 0))
            writer({"type": "profile", "resource_type": "expert", "data": profile})

    return {"final_profiles": new_profiles, "seen_urls": list(seen), "best_score": best}


def should_self_correct(state: FindProfilesState) -> str:
    if state["pass_num"] == 1 and state["best_score"] < 700:
        return "self_correct"
    return "summarize"


def prepare_pass_2_node(state: FindProfilesState) -> Dict[str, Any]:
    failure_reasons = [p.get("llm_reason") for p in state["final_profiles"] if p.get("llm_score", 0) > 0][:3]
    feedback = f"Top scores were low ({state['best_score']}/1000). Reasons: {' | '.join(failure_reasons)}"
    return {"pass_num": 2, "feedback": feedback}


async def summarize_node(state: FindProfilesState) -> Dict[str, Any]:
    writer = get_stream_writer()
    writer({"type": "status", "message": "Drafting executive summary..."})
    sorted_profiles = sorted(
        state["final_profiles"],
        key=lambda x: (x.get("llm_score", 0), x.get("popularity", 0)),
        reverse=True,
    )
    summary = await generate_search_summary(state["description"], state["all_queries"], sorted_profiles)
    writer({"type": "done", "summary": summary, "total": len(sorted_profiles)})
    return {}


_find_g = StateGraph(FindProfilesState)
_find_g.add_node("generate_queries", generate_queries_node)
_find_g.add_node("serp_search", serp_search_node)
_find_g.add_node("process_profiles", process_profiles_node)
_find_g.add_node("prepare_pass_2", prepare_pass_2_node)
_find_g.add_node("summarize", summarize_node)
_find_g.add_edge(START, "generate_queries")
_find_g.add_edge("generate_queries", "serp_search")
_find_g.add_edge("serp_search", "process_profiles")
_find_g.add_conditional_edges(
    "process_profiles",
    should_self_correct,
    {"self_correct": "prepare_pass_2", "summarize": "summarize"},
)
_find_g.add_edge("prepare_pass_2", "generate_queries")
_find_g.add_edge("summarize", END)
find_profiles_graph = _find_g.compile()


# ============================================================
# LangGraph #3: Unified solutions search (experts + tools + businesses)
# ============================================================

class FindSolutionsState(TypedDict):
    description: str

    # Classification output (set by classify_needs_node)
    needs_experts:     bool
    needs_tools:       bool
    needs_businesses:  bool
    expert_focus:      str
    tool_focus:        str
    business_focus:    str

    # Per-branch query + result accumulation
    expert_queries:    Annotated[List[str], operator.add]
    tool_queries:      Annotated[List[str], operator.add]
    business_queries:  Annotated[List[str], operator.add]

    expert_results:    Annotated[List[dict], operator.add]
    tool_results:      Annotated[List[dict], operator.add]
    business_results:  Annotated[List[dict], operator.add]

    stack_summary:     str


# --- Classification node ----------------------------------------------------

async def classify_needs_node(state: FindSolutionsState) -> Dict[str, Any]:
    writer = get_stream_writer()
    writer({"type": "status", "message": "Classifying solution types needed..."})

    prompt = f"""
    Analyze this problem and decide which kinds of solutions would help.

    Problem: "{state['description']}"

    For each of the three solution types, decide if it's worth searching.
    Be inclusive but not wasteful — only mark False if a type is clearly irrelevant.

    Examples:
      - "I need session replay" → needs_tools=True, others maybe False
      - "I need a board advisor" → needs_experts=True, others False
      - "Our onboarding is broken" → all three likely True (expert to diagnose, tool to measure, agency to redesign)
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format=SolutionClassification,
    )
    cls = completion.choices[0].message.parsed

    # Experts are core — always run that branch regardless of classifier.
    final_needs_experts    = True
    final_needs_tools      = bool(cls.needs_tools)
    final_needs_businesses = bool(cls.needs_businesses)

    logger.info(
        f"Classifier raw: experts={cls.needs_experts}, tools={cls.needs_tools}, "
        f"businesses={cls.needs_businesses}. Reasoning: {cls.reasoning}"
    )
    logger.info(
        f"Branches that will run: experts={final_needs_experts}, "
        f"tools={final_needs_tools}, businesses={final_needs_businesses}"
    )

    writer({"type": "classification", "data": {
        "needs_experts":    final_needs_experts,
        "needs_tools":      final_needs_tools,
        "needs_businesses": final_needs_businesses,
        "reasoning":        cls.reasoning,
    }})

    return {
        "needs_experts":    final_needs_experts,
        "needs_tools":      final_needs_tools,
        "needs_businesses": final_needs_businesses,
        "expert_focus":     cls.expert_focus,
        "tool_focus":       cls.tool_focus,
        "business_focus":   cls.business_focus,
    }


# --- Experts branch ---------------------------------------------------------

async def experts_branch_node(state: FindSolutionsState) -> Dict[str, Any]:
    writer = get_stream_writer()
    if not state.get("needs_experts"):
        logger.warning("[Experts] Branch SKIPPED — needs_experts=False.")
        writer({"type": "status", "message": "[Experts] Skipped (classifier said not needed)."})
        return {}

    try:
        writer({"type": "status", "message": "[Experts] Generating queries..."})

        queries = await generate_search_queries(state["description"])
        logger.info(f"[Experts] Generated {len(queries)} queries.")
        writer({"type": "queries", "resource_type": "expert", "data": queries})

        results_nested = await asyncio.gather(
            *(fetch_serpapi_results(q) for q in queries),
            return_exceptions=True,
        )
        all_links_set: set = set()
        for sub in results_nested:
            if isinstance(sub, Exception):
                logger.warning(f"[Experts] SerpAPI sub-task failed: {sub}")
                continue
            for url in sub:
                all_links_set.add(url)
        all_links = sorted(all_links_set)

        logger.info(f"[Experts] {len(all_links)} unique LinkedIn URLs to process.")
        writer({"type": "found_total", "resource_type": "expert", "total": len(all_links)})

        if not all_links:
            logger.warning("[Experts] No LinkedIn URLs found. SerpAPI may be rate-limited or queries returned nothing.")
            writer({"type": "status", "message": "[Experts] No LinkedIn URLs found this pass."})
            return {"expert_queries": queries, "expert_results": []}

        seen: set = set()
        new_profiles: List[dict] = []
        tasks = [_process_single_profile(link, state["description"]) for link in all_links]
        for coro in asyncio.as_completed(tasks):
            try:
                profile = await coro
            except Exception as e:
                logger.warning(f"[Experts] _process_single_profile crashed: {e}")
                continue
            if profile and profile["profile_url"] not in seen:
                seen.add(profile["profile_url"])
                new_profiles.append(profile)
                writer({"type": "profile", "resource_type": "expert", "data": profile})

        logger.info(f"[Experts] Branch finished. {len(new_profiles)} profiles found.")
        return {"expert_queries": queries, "expert_results": new_profiles}

    except Exception as e:
        # A branch exception used to kill the whole graph, dropping the stream.
        # Now: log it, emit a status event, and return empty so other branches
        # + synthesis still run.
        logger.exception(f"[Experts] Branch crashed: {e}")
        try:
            writer({
                "type": "error",
                "resource_type": "expert",
                "message": f"Experts branch failed: {str(e)[:160]}",
            })
        except Exception:
            pass
        return {"expert_queries": [], "expert_results": []}


# --- Tools branch ----------------------------------------------------------

async def tools_branch_node(state: FindSolutionsState) -> Dict[str, Any]:
    if not state.get("needs_tools"):
        return {}
    writer = get_stream_writer()

    try:
        writer({"type": "status", "message": "[Tools] Generating queries..."})

        queries = await generate_tool_queries(state["description"], state.get("tool_focus", ""))
        writer({"type": "queries", "resource_type": "tool", "data": queries})

        results_nested = await asyncio.gather(
            *(fetch_serpapi_results(q, profile_filter=False) for q in queries),
            return_exceptions=True,
        )

        seen_urls: set = set()
        flat: List[dict] = []
        for sub in results_nested:
            if isinstance(sub, Exception):
                logger.warning(f"[Tools] SerpAPI sub-task failed: {sub}")
                continue
            for item in sub:
                url = item.get("link")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                flat.append(item)
        writer({"type": "found_total", "resource_type": "tool", "total": len(flat)})

        flat = flat[:60]
        new_tools: List[dict] = []
        seen_names: set = set()
        tasks = [_process_single_tool(item, state["description"]) for item in flat]
        for coro in asyncio.as_completed(tasks):
            try:
                tool = await coro
            except Exception as e:
                logger.warning(f"[Tools] _process_single_tool crashed: {e}")
                continue
            if not tool:
                continue
            name_key = (tool.get("tool_name") or "").lower().strip()
            if name_key and name_key in seen_names:
                continue
            if name_key:
                seen_names.add(name_key)
            new_tools.append(tool)
            writer({"type": "profile", "resource_type": "tool", "data": tool})

        return {"tool_queries": queries, "tool_results": new_tools}

    except Exception as e:
        logger.exception(f"[Tools] Branch crashed: {e}")
        try:
            writer({
                "type": "error",
                "resource_type": "tool",
                "message": f"Tools branch failed: {str(e)[:160]}",
            })
        except Exception:
            pass
        return {"tool_queries": [], "tool_results": []}


# --- Businesses branch -----------------------------------------------------

async def businesses_branch_node(state: FindSolutionsState) -> Dict[str, Any]:
    if not state.get("needs_businesses"):
        return {}
    writer = get_stream_writer()

    try:
        writer({"type": "status", "message": "[Agencies] Generating queries..."})

        queries = await generate_business_queries(state["description"], state.get("business_focus", ""))
        writer({"type": "queries", "resource_type": "business", "data": queries})

        results_nested = await asyncio.gather(
            *(fetch_serpapi_results(q, profile_filter=False) for q in queries),
            return_exceptions=True,
        )

        seen_urls: set = set()
        flat: List[dict] = []
        for sub in results_nested:
            if isinstance(sub, Exception):
                logger.warning(f"[Agencies] SerpAPI sub-task failed: {sub}")
                continue
            for item in sub:
                url = item.get("link")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                flat.append(item)
        writer({"type": "found_total", "resource_type": "business", "total": len(flat)})

        flat = flat[:60]
        new_firms: List[dict] = []
        seen_names: set = set()
        tasks = [_process_single_business(item, state["description"]) for item in flat]
        for coro in asyncio.as_completed(tasks):
            try:
                firm = await coro
            except Exception as e:
                logger.warning(f"[Agencies] _process_single_business crashed: {e}")
                continue
            if not firm:
                continue
            name_key = (firm.get("firm_name") or "").lower().strip()
            if name_key and name_key in seen_names:
                continue
            if name_key:
                seen_names.add(name_key)
            new_firms.append(firm)
            writer({"type": "profile", "resource_type": "business", "data": firm})

        return {"business_queries": queries, "business_results": new_firms}

    except Exception as e:
        logger.exception(f"[Agencies] Branch crashed: {e}")
        try:
            writer({
                "type": "error",
                "resource_type": "business",
                "message": f"Agencies branch failed: {str(e)[:160]}",
            })
        except Exception:
            pass
        return {"business_queries": [], "business_results": []}


# --- Synthesis: produce the "Solution Stack" recommendation ----------------

async def synthesize_stack_node(state: FindSolutionsState) -> Dict[str, Any]:
    writer = get_stream_writer()
    writer({"type": "status", "message": "Synthesizing your Solution Stack..."})

    def top(items, n=3):
        return sorted(items, key=lambda x: x.get("llm_score", 0), reverse=True)[:n]

    top_experts    = top(state.get("expert_results") or [])
    top_tools      = top(state.get("tool_results") or [])
    top_businesses = top(state.get("business_results") or [])

    def fmt_expert(p):
        name = extract_name_for_llm(p)
        return f"- {name} ({p.get('dynamic_title')}): {p.get('llm_reason')}"
    def fmt_tool(p):
        return f"- {p.get('tool_name')} ({p.get('dynamic_title')}): {p.get('llm_reason')}"
    def fmt_biz(p):
        return f"- {p.get('firm_name')} ({p.get('dynamic_title')}): {p.get('llm_reason')}"

    prompt = f"""
    You are a strategic consultant. Produce a "Solution Stack" recommendation for the user's problem.

    Problem: {state['description']}

    Top experts found:
    {chr(10).join(fmt_expert(p) for p in top_experts) or "(none)"}

    Top tools found:
    {chr(10).join(fmt_tool(p) for p in top_tools) or "(none)"}

    Top agencies/firms found:
    {chr(10).join(fmt_biz(p) for p in top_businesses) or "(none)"}

    Write a concise, well-structured recommendation that:
    1. Reframes the problem in one sentence.
    2. Recommends a SPECIFIC combination of resources (e.g., "Buy Tool X, hire Expert Y for 2 weeks,
       skip the agency unless you have $50k+").
    3. Includes trade-offs by budget tier: Lean ($) / Standard ($$) / Premium ($$$).
    4. Uses clean HTML only (<h3>, <p>, <ul>, <li>, <strong>). No markdown, no <html>/<body> tags.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        stack_html = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Stack synthesis failed: {e}")
        stack_html = "<p>Synthesis failed; please review individual results.</p>"

    totals = {
        "experts":    len(state.get("expert_results") or []),
        "tools":      len(state.get("tool_results") or []),
        "businesses": len(state.get("business_results") or []),
    }
    writer({"type": "done", "stack_html": stack_html, "totals": totals})
    return {"stack_summary": stack_html}


# --- Wire the graph -------------------------------------------------------

_sol_g = StateGraph(FindSolutionsState)
_sol_g.add_node("classify_needs",       classify_needs_node)
_sol_g.add_node("experts_branch",       experts_branch_node)
_sol_g.add_node("tools_branch",         tools_branch_node)
_sol_g.add_node("businesses_branch",    businesses_branch_node)
_sol_g.add_node("synthesize_stack",     synthesize_stack_node)

_sol_g.add_edge(START, "classify_needs")
_sol_g.add_edge("classify_needs", "experts_branch")
_sol_g.add_edge("classify_needs", "tools_branch")
_sol_g.add_edge("classify_needs", "businesses_branch")
_sol_g.add_edge("experts_branch",    "synthesize_stack")
_sol_g.add_edge("tools_branch",      "synthesize_stack")
_sol_g.add_edge("businesses_branch", "synthesize_stack")
_sol_g.add_edge("synthesize_stack", END)

find_solutions_graph = _sol_g.compile()


# ============================================================
# LangGraph #4: Candidate Comparison (HOLISTIC)
# ============================================================

class CompareGraphState(TypedDict):
    description: str
    profiles: List[Dict[str, Any]]
    other_resources: Optional[Dict[str, List[Dict[str, Any]]]]
    comparison_html: str


def _format_other_resources_for_prompt(other: Optional[Dict[str, List[Dict[str, Any]]]]) -> Tuple[str, str]:
    """Render top tools + agencies from the same search into prompt-friendly text blocks."""
    if not other:
        return "(none surfaced)", "(none surfaced)"

    def _sorted_top(items: List[Dict[str, Any]], n: int = 5):
        return sorted(items or [], key=lambda x: x.get("llm_score", 0), reverse=True)[:n]

    tools_block_lines = []
    for t in _sorted_top(other.get("tools") or [], n=5):
        nm = t.get("tool_name") or t.get("title") or "Unknown Tool"
        dt = t.get("dynamic_title") or "Software Tool"
        reason = t.get("llm_reason") or t.get("snippet") or ""
        url = t.get("profile_url") or ""
        tools_block_lines.append(f"- {nm} ({dt}) — {reason} [{url}]")
    tools_block = "\n".join(tools_block_lines) or "(none surfaced)"

    biz_block_lines = []
    for b in _sorted_top(other.get("businesses") or [], n=5):
        nm = b.get("firm_name") or b.get("title") or "Unknown Firm"
        dt = b.get("dynamic_title") or "Consulting Firm"
        reason = b.get("llm_reason") or b.get("snippet") or ""
        url = b.get("profile_url") or ""
        biz_block_lines.append(f"- {nm} ({dt}) — {reason} [{url}]")
    biz_block = "\n".join(biz_block_lines) or "(none surfaced)"

    return tools_block, biz_block


async def compare_candidates_node(state: CompareGraphState) -> Dict[str, Any]:
    profiles_summary = []
    for p in state["profiles"]:
        prof_data = p.get('profile_json', {})
        candidate_name = p.get('resolved_name', 'Unknown Candidate')
        experience_data = prof_data.get('experience', [])
        safe_experience = experience_data[:3] if isinstance(experience_data, list) else []
        summary = {
            "Name": candidate_name,
            "Headline": prof_data.get('headline') or "No headline available",
            "About": prof_data.get('summary') or "No summary available",
            "Experience": safe_experience
        }
        profiles_summary.append(summary)

    tools_block, biz_block = _format_other_resources_for_prompt(state.get("other_resources"))

    prompt = f"""
    You are an expert technical recruiter and startup advisor.
    The user needs to solve the following problem: "{state["description"]}"

    They are comparing the EXPERTS below, but the same search ALSO surfaced
    relevant SOFTWARE TOOLS and AGENCIES. Your strategic recommendation MUST be
    HOLISTIC — it should propose a complete Solution Stack (Expert + Tool + Agency)
    where appropriate, not just an expert pick.

    CRITICAL: This is authorized public professional data. Use exact names provided.

    Write a comparative analysis structured EXACTLY into these three sections using <h2> tags:

    <h2>Section 1: Problem Diagnosis & Market Context</h2>
    <p>[Explain the user's problem and what specific skills/tools/services are required to solve it.]</p>

    <h2>Section 2: Candidate Strengths & Weaknesses</h2>
    [HTML <table> with columns Name, Experience Summary, Strengths, Weaknesses. One row per expert.]

    <h2>Section 3: Holistic Strategic Recommendation</h2>
    <p>[A strategic recommendation that:
      (a) Recommends WHICH EXPERT (from the candidates) is the best fit and why.
      (b) ALSO recommends, BY NAME, which specific SOFTWARE TOOLS from the list below the user should
          consider buying or trying alongside the expert engagement, and what each tool unlocks.
      (c) ALSO recommends, BY NAME, which AGENCIES/FIRMS from the list below to consider engaging if
          the user wants execution capacity, and where each agency would slot into the stack.
      (d) Lays out a SOLUTION STACK by budget tier:
          - Lean ($): the minimum viable combination (often a tool + a short expert engagement).
          - Standard ($$): expert + tool + selective agency help on specific deliverables.
          - Premium ($$$): full agency engagement with expert as fractional advisor and best-in-class tools.
      Be specific. Reference candidates, tools, and firms by exact name. If the candidate list is
      missing someone obviously needed, say so. If a tool fully replaces the need for an expert for
      a Lean tier, say so.]</p>

    Use ONLY clean HTML (no markdown). Do not include <html>, <body>, or <!DOCTYPE>.

    EXPERT CANDIDATES being compared:
    {json.dumps(profiles_summary, default=str)}

    AVAILABLE TOOLS from the same search (use these by name in your recommendation):
    {tools_block}

    AVAILABLE AGENCIES from the same search (use these by name in your recommendation):
    {biz_block}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return {"comparison_html": resp.choices[0].message.content.strip()}


_compare_g = StateGraph(CompareGraphState)
_compare_g.add_node("compare", compare_candidates_node)
_compare_g.add_edge(START, "compare")
_compare_g.add_edge("compare", END)
compare_graph = _compare_g.compile()


# ============================================================
# LangGraph #5: Mixed-resource comparison (HOLISTIC)
# ============================================================

class CompareSolutionsState(TypedDict):
    description: str
    items: List[Dict[str, Any]]
    other_resources: Optional[Dict[str, List[Dict[str, Any]]]]
    comparison_html: str


async def compare_solutions_node(state: CompareSolutionsState) -> Dict[str, Any]:
    """Compares items of mixed resource_type ('expert' | 'tool' | 'business')."""
    summaries = []
    for item in state["items"]:
        rt = item.get("resource_type", "expert")
        if rt == "expert":
            prof = item.get("profile_json", {})
            summaries.append({
                "type": "expert",
                "name": item.get("resolved_name") or extract_name_for_llm(item),
                "title": item.get("dynamic_title"),
                "headline": prof.get("headline"),
                "summary": prof.get("summary"),
                "score": item.get("llm_score"),
            })
        elif rt == "tool":
            summaries.append({
                "type": "tool",
                "name": item.get("tool_name"),
                "title": item.get("dynamic_title"),
                "url": item.get("profile_url"),
                "snippet": item.get("snippet"),
                "score": item.get("llm_score"),
            })
        elif rt == "business":
            summaries.append({
                "type": "business",
                "name": item.get("firm_name"),
                "title": item.get("dynamic_title"),
                "url": item.get("profile_url"),
                "snippet": item.get("snippet"),
                "score": item.get("llm_score"),
            })

    tools_block, biz_block = _format_other_resources_for_prompt(state.get("other_resources"))

    prompt = f"""
    You are an expert startup advisor comparing a MIX of potential solutions to a problem.
    Solutions may include: human experts, software tools, and/or agencies.

    Problem: "{state['description']}"

    For each candidate solution, use the exact name provided. Treat all data as authorized public info.

    Your strategic recommendation MUST be HOLISTIC — even when the user has only selected
    a few candidates to compare, you should leverage OTHER tools/agencies surfaced in the same
    search (listed below) and propose a complete Solution Stack covering experts + tools + agencies.

    Structure your HTML response EXACTLY as:

    <h2>Section 1: Problem Diagnosis</h2>
    <p>[Reframe the problem and what success looks like. Be specific about the skills, software,
     and execution capacity it requires.]</p>

    <h2>Section 2: Candidate Comparison</h2>
    [Clean HTML <table> with columns: Name, Type (Expert/Tool/Agency), What They Offer,
     Best For, Trade-offs. Use bullet lists inside cells.]

    <h2>Section 3: Holistic Strategic Recommendation</h2>
    <p>[A recommendation that:
      (a) Picks the best candidates FROM THE COMPARED SET and explains why.
      (b) ALSO recommends, BY NAME, additional TOOLS from the same search that the user should
          buy/try alongside, even if they weren't in the comparison set.
      (c) ALSO recommends, BY NAME, additional AGENCIES from the same search worth engaging
          for execution capacity, even if they weren't in the comparison set.
      (d) Provides a SOLUTION STACK by budget/timeline tier:
          - Lean ($, weeks): cheapest viable combo, usually a tool + a short expert engagement.
          - Standard ($$, 1-3 months): expert + tool + targeted agency help.
          - Premium ($$$, 3-6 months): full agency engagement with expert as advisor and premium tools.
      Be explicit about WHICH names go in WHICH tier. Reference candidates, tools, and firms
      by exact name. If a tool fully obviates the need for an expert at the Lean tier, say so.
      If an agency would only make sense above a certain budget, say so.]</p>

    Use ONLY clean HTML (<h2>, <h3>, <p>, <ul>, <li>, <strong>, <table>, <tr>, <th>, <td>).
    No markdown. No <html>, <body>, or <!DOCTYPE>.

    CANDIDATES being compared:
    {json.dumps(summaries, default=str)}

    ADDITIONAL TOOLS surfaced in the same search (use by name where helpful):
    {tools_block}

    ADDITIONAL AGENCIES surfaced in the same search (use by name where helpful):
    {biz_block}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return {"comparison_html": resp.choices[0].message.content.strip()}


_compare_sol_g = StateGraph(CompareSolutionsState)
_compare_sol_g.add_node("compare", compare_solutions_node)
_compare_sol_g.add_edge(START, "compare")
_compare_sol_g.add_edge("compare", END)
compare_solutions_graph = _compare_sol_g.compile()


# ============================================================
# FastAPI Endpoints
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):    # ← add ": Request"
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

# --- Feature 1: Chat Diagnostician ---
@app.post("/chat_gather_info")
async def chat_gather_info(req: ChatRequest):
    try:
        out = await chat_graph.ainvoke({"messages": req.messages, "result": None})
        return out["result"]
    except Exception as e:
        logger.error(f"Chat agent error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat.")


# --- Feature 2a: LEGACY experts-only streaming endpoint (backwards compat) ---
@app.post("/find_profiles_agentic")
async def find_profiles_agentic(req: JobRequest):
    if not req.description or not req.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty.")

    initial_state: FindProfilesState = {
        "description": req.description.strip(),
        "pass_num": 1,
        "feedback": None,
        "pass_queries": [],
        "all_queries": [],
        "pass_links": [],
        "seen_urls": [],
        "final_profiles": [],
        "best_score": 0,
    }

    async def stream_generator():
        async for chunk in find_profiles_graph.astream(initial_state, stream_mode="custom"):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


# --- Feature 2b: Unified solutions endpoint (experts + tools + businesses) ---
@app.post("/find_solutions_agentic")
async def find_solutions_agentic(req: JobRequest):
    """
    Streams events in real-time. Each event is one of:
      {"type": "status", "message": "..."}
      {"type": "classification", "data": {...}}
      {"type": "queries", "resource_type": "expert|tool|business", "data": [...]}
      {"type": "found_total", "resource_type": "...", "total": N}
      {"type": "profile", "resource_type": "expert|tool|business", "data": {...}}
      {"type": "error", "resource_type": "...", "message": "..."}  (non-fatal branch failure)
      {"type": "done", "stack_html": "...", "totals": {...}}

    Every call is a fresh search. The frontend handles refinements by building
    an updated, self-contained `description` from the full chat history and
    POSTing it here — no continuation state is tracked server-side.
    """
    if not req.description or not req.description.strip():
        logger.error("/find_solutions_agentic called with empty/null description.")
        raise HTTPException(
            status_code=400,
            detail="Description cannot be empty. Please complete the diagnosis chat first."
        )

    initial_state: FindSolutionsState = {
        "description":      req.description.strip(),
        "needs_experts":    False,
        "needs_tools":      False,
        "needs_businesses": False,
        "expert_focus":     "",
        "tool_focus":       "",
        "business_focus":   "",
        "expert_queries":   [],
        "tool_queries":     [],
        "business_queries": [],
        "expert_results":   [],
        "tool_results":     [],
        "business_results": [],
        "stack_summary":    "",
    }

    async def stream_generator():
        emitted_done = False
        try:
            async for chunk in find_solutions_graph.astream(initial_state, stream_mode="custom"):
                yield json.dumps(chunk) + "\n"
                if isinstance(chunk, dict) and chunk.get("type") == "done":
                    emitted_done = True
        except Exception as e:
            # Any unhandled graph exception used to drop the connection,
            # making the frontend think the search "failed" and wipe partial
            # results. Now we emit an explicit error event + a fallback done so
            # the client can render the partial results gracefully.
            logger.exception(f"Graph stream crashed: {e}")
            try:
                yield json.dumps({
                    "type": "error",
                    "message": f"Stream interrupted: {str(e)[:200]}",
                }) + "\n"
            except Exception:
                pass
        finally:
            if not emitted_done:
                try:
                    yield json.dumps({
                        "type": "done",
                        "stack_html": "<p style='color:#ef4444'><strong>Search ended early.</strong> Some branches did not complete. The results below are what was collected so far.</p>",
                        "totals": {},
                    }) + "\n"
                except Exception:
                    pass

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


# --- Feature 3: Audio Transcription ---
@app.post("/transcribe")
def transcribe_audio(audio: UploadFile = File(...)):
    temp_audio_path = f"temp_{uuid.uuid4().hex}.webm"
    try:
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(audio.file.read())
        with open(temp_audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return {"text": transcription.text}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed.")
    finally:
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to delete temp audio: {cleanup_error}")


# --- Feature 4a: Legacy expert-only comparison (HOLISTIC) ---
@app.post("/compare_candidates")
async def compare_candidates(req: CompareRequest):
    if len(req.profiles) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 profiles to compare.")
    try:
        out = await compare_graph.ainvoke({
            "description": req.description,
            "profiles": req.profiles,
            "other_resources": req.other_resources or {},
            "comparison_html": "",
        })
        return {"comparison_html": out["comparison_html"]}
    except Exception as e:
        logger.error(f"Failed to generate comparison: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate comparison.")


# --- Feature 4b: Mixed-resource comparison (HOLISTIC) ---
@app.post("/compare_solutions")
async def compare_solutions(req: CompareSolutionsRequest):
    if len(req.items) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 items to compare.")
    try:
        out = await compare_solutions_graph.ainvoke({
            "description": req.description,
            "items": req.items,
            "other_resources": req.other_resources or {},
            "comparison_html": "",
        })
        return {"comparison_html": out["comparison_html"]}
    except Exception as e:
        logger.error(f"Failed to generate solution comparison: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate comparison.")


# --- Feature 5: PDF Export ---
@app.post("/download_report_pdf")
async def download_report_pdf(req: PDFRequest):
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
            body {{ font-family: 'Inter', sans-serif; color: #0f172a; line-height: 1.6; padding: 20px; }}
            h2 {{ color: #2ab0eb; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; margin-top: 30px; }}
            h3 {{ color: #0f172a; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border-bottom: 1px solid #e2e8f0; padding: 12px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f8fafc; color: #2ab0eb; text-transform: uppercase; font-size: 12px; }}
            li {{ margin-bottom: 8px; }}
        </style>
    </head>
    <body>
        <h2>📑 SolverSearch Solution Analysis</h2>
        {req.html_content}
    </body>
    </html>
    """
    try:
        pdf_bytes = HTML(string=full_html).write_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=SolverSearch_Analysis.pdf"}
        )
    except Exception as e:
        logger.error(f"PDF Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate PDF.")


if __name__ == "__main__":
    uvicorn.run("solver_agentic10:app", host="0.0.0.0", port=8000, reload=True)
