"""
============================================================
SolverSearch Backend (FastAPI)
============================================================
Run:
    python solver.py
"""
from fastapi import UploadFile, File
import os
import uuid
import os, asyncio, httpx, re, logging, json
from urllib.parse import urlparse, urlunparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import uvicorn
from typing import Any, Dict, List, Optional, Tuple, Literal
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from weasyprint import HTML, CSS

# ------------------------------ Configuration ------------------------------
OPENAI_API_KEY = "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" #"sk-TI4QAzumMX3a_nyzjV7JLGaJxrfBd2ZXUWHW2s8mf6T3BlbkFJHdavfukK1JF6Y3EgoCj7CuhN9wQSwEAOXgI9lDKA4A" #"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"
SERP_API_KEY   = "6fac0d40369a7711c0381f3c9ed349ec7e63dda084ec9c93eef768737770b826"
SCRAPINGDOG_API_KEY = "693af892674e92afaa2fdc32"

NUM_QUERIES = 2
NUM_RESULTS_PER_QUERY = 3

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="SolverSearch Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solversearch")

SERP_CONCURRENCY = int(os.getenv("SERP_CONCURRENCY", "3"))
_serp_sem = asyncio.Semaphore(SERP_CONCURRENCY)
SCRAPE_CONCURRENCY = int(os.getenv("SCRAPE_CONCURRENCY", "1"))
_scrape_sem = asyncio.Semaphore(SCRAPE_CONCURRENCY)

# ------------------------------ Schemas ------------------------------
class PDFRequest(BaseModel):
    html_content: str

class JobRequest(BaseModel):
    description: str

class ChatMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class CompareRequest(BaseModel):
    description: str
    profiles: List[Dict[str, Any]]

# --- NEW: Structured Output Schema for Chat Options ---
class ChatTurnResponse(BaseModel):
    status: Literal["chatting", "complete"] = Field(
        description="Return 'chatting' if you need more information, or 'complete' if the problem is fully diagnosed."
    )
    reply: Optional[str] = Field(
        description="Your conversational response to the user. Required if status is 'chatting'."
    )
    options: Optional[List[str]] = Field(
        description="2 to 4 logical, highly relevant quick-reply options. Make them short (1-4 words). DO NOT include 'Other', 'Not sure', or 'None'. Required if status is 'chatting'.",
        default=[]
    )
    final_description: Optional[str] = Field(
        description="The comprehensive, finalized problem description. Required ONLY if status is 'complete'."
    )

# ------------------------------ Feature 1: Chat Diagnostician ------------------------------
@app.post("/chat_gather_info")
async def chat_gather_info(req: ChatRequest):
    """
    Acts as a startup advisor/diagnostician. It reviews the conversation history.
    If it needs more info about the problem, it asks clarifying questions.
    Once the threshold of context is met, it outputs the ideal expert profile.
    """
    system_prompt = """
    You are the SolverSearch AI, an expert diagnostician and advisor for startup founders and entrepreneurs.
    Users will come to you with vague business, product, growth, or operational problems (e.g., "Users keep dropping off after onboarding").
    Your goal is NOT to ask for a job title. Your goal is to ask clarifying questions to diagnose the root cause so you can identify the EXACT type of expert, consultant, or advisor they need.

    Review the conversation history.
    - If you lack context, ask 1 or 2 concise clarifying questions. Focus on identifying: 
      1. The domain/type of product (e.g., B2B SaaS, mobile app).
      2. Their current state/data maturity (e.g., "Do you track analytics?", "Have you run interviews?").
      3. Specific symptoms (e.g., "Do they drop off instantly or after a few days?").
    
    - When you ask a clarifying question, you MUST also provide 2 to 4 highly logical, clickable quick-reply options. 
      These options should represent the most common or likely answers to your specific question to save the user from typing.
      
      Guidelines for options:
      - Keep them punchy and actionable (e.g., "B2B SaaS", "E-commerce", "Under $5,000", "Enterprise Level").
      - Ensure they directly answer the question you just asked.
      - DO NOT include options like "Not sure", "Other", or "None". The frontend UI natively tells the user to type their specific constraints if the buttons don't match.

    - Once you have enough signal (domain + symptom + current state), summarize the problem and output a comprehensive description of the IDEAL EXPERT needed to solve this problem (e.g., "A Growth Product Manager specializing in B2B SaaS activation and user interviews...").
    """
    
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in req.messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    try:
        # Use .parse with the Pydantic model to guarantee structured JSON with the options array
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=api_messages,
            temperature=0.6,
            response_format=ChatTurnResponse
        )
        
        llm_response = completion.choices[0].message.parsed
        
        if llm_response.status == "chatting":
            return {
                "status": "chatting",
                "reply": llm_response.reply,
                "options": llm_response.options
            }
        else:
            return {
                "status": "complete",
                "final_description": llm_response.final_description
            }
            
    except Exception as e:
        logger.error(f"Chat agent error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat.")

# ------------------------------ GPT Query Generation (Agentic) ------------------------------
async def generate_search_queries(description: str, feedback: str = None) -> list[str]:
    """Generates queries aimed at finding problem-solvers, freelancers, or experts."""
    prompt = f"""
    Generate {NUM_QUERIES} highly effective Google search queries to find LinkedIn profiles (using site:linkedin.com/in/) of EXPERTS, ADVISORS, or CONSULTANTS who can solve the following problem:
    
    "{description}"
    
    Focus on keywords related to the specific skills, domain (e.g., B2B SaaS), and problem-solving experience. Include terms like "Advisor", "Consultant", "Freelance", or specific senior titles if applicable.
    """
    
    if feedback:
        prompt += f"\n\nWARNING - Previous search failed with this feedback: {feedback}\n"
        prompt += "Please ADJUST your search strategy. Use different keywords, synonyms, or look for different job titles that might solve this problem."
        
    prompt += "\nReturn only the queries, one per line."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7 if feedback else 0.4,
    )
    text = resp.choices[0].message.content.strip()
    return [q.strip() for q in text.splitlines() if q.strip()]

# ------------------------------ Core Processing Functions ------------------------------
PROFILE_RE = re.compile(r"https?://(?:[\w-]+\.)*linkedin\.com/(?:in|pub)/[^/?#]+/?", re.IGNORECASE)

def _canonicalize(url: str) -> str | None:
    m = PROFILE_RE.match(url)
    if not m: return None
    u = urlparse(m.group(0))
    path = u.path if u.path.endswith("/") else u.path + "/"
    return urlunparse((u.scheme, u.netloc.lower(), path, "", "", ""))

async def fetch_serpapi_results(query: str) -> list[str]:
    serp_url = "https://serpapi.com/search.json"
    params = {
        "engine": "google", "q": query, "num": NUM_RESULTS_PER_QUERY,
        "api_key": SERP_API_KEY, "hl": "en", "gl": "us", "google_domain": "google.com",
    }
    async with _serp_sem:
        async with httpx.AsyncClient(timeout=130.0) as ac:
            resp = await ac.get(serp_url, params=params)

    if resp.status_code != 200:
        return []
    
    data = resp.json()
    urls = set()
    for item in data.get("organic_results", []):
        if link := item.get("link"):
            if canon := _canonicalize(link): urls.add(canon)
    return list(urls)

async def scrape_profile(profile_url: str) -> Dict[str, Any]:
    try:
        profile_id = urlparse(profile_url).path.strip("/").split('/')[-1]
    except:
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
                
                # Check if the API returned a list or a dictionary
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

def extract_popularity(profile_json: Dict[str, Any]) -> int:
    candidates = []
    for key in ("followers", "connections", "followers_count"):
        v = profile_json.get(key)
        try:
            if isinstance(v, str): v = int(re.sub(r"[^0-9]", "", v) or 0)
            elif isinstance(v, (int, float)): v = int(v)
            if v and v > 0: candidates.append(v)
        except: pass
    return max(candidates) if candidates else 0


async def score_profile_with_llm(description: str, profile_json: Dict[str, Any]) -> Tuple[int, str, str]:
    prompt = (
        "Score how well this expert's LinkedIn profile proves they can SOLVE the problem described below. "
        "Rate strictly on a scale from 1 to 1000 (e.g., 850, 920). Do NOT use a 1-10 scale.\n"
        "Also, generate a 3-5 word 'dynamic_title' summarizing their specific expertise related to the problem (e.g., 'B2B SaaS Growth Consultant', 'Data Pipeline Architect').\n"
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
        
        # Return the score, reason, and the new dynamic title
        return (
            min(max(int(parsed.get('score', 1)), 1), 1000), 
            parsed.get('reason', ''),
            parsed.get('dynamic_title', 'Specialized Expert')
        )
    except:
        return 0, "Failed to parse LLM score.", "Expert / Consultant"

async def generate_search_summary(description: str, queries: List[str], top_profiles: List[Dict]) -> str:
    """Generates a cohesive executive summary of the search results."""
    top_candidates = "\n".join(
        [f"- {p['profile_json'].get('full_name', 'Unknown')}: {p.get('llm_reason')}" 
         for p in top_profiles[:3] if p.get('llm_reason')]
    )
    
    prompt = f"""
    You are an AI Search Coordinator. Write a brief, professional 2-3 sentence executive summary of the search operation you just completed.
    
    Diagnosed Problem: {description}
    Search Queries Used: {', '.join(queries)}
    Top Candidates Evaluated:
    {top_candidates}
    
    Your summary should explain what kind of experts were successfully found to solve the user's specific problem based on the evaluation data. Do not use bullet points, just a short cohesive paragraph.
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
        return "Successfully executed agentic search and evaluated expert profiles based on the diagnosis."


async def _process_single_profile(link: str, description: str) -> dict:
    """Scrapes and scores a single profile, returning the complete dictionary."""
    prof_json = await scrape_profile(link)
    if not prof_json:
        return None
    
    # Catch the 3 variables now
    score, reason, dynamic_title = await score_profile_with_llm(description, prof_json)
    
    return {
        "profile_url": prof_json.get("_profile_url"),
        "popularity": extract_popularity(prof_json),
        "llm_score": score,
        "llm_reason": reason,
        "dynamic_title": dynamic_title, # <-- Add this line
        "profile_json": prof_json,
    }

# ------------------------------ Feature 2: Agentic Search Loop (Streaming) ------------------------------
@app.post("/find_profiles_agentic")
async def find_profiles_agentic(req: JobRequest):
    """
    Streams status updates and scraped/scored profiles back to the client in real-time
    using NDJSON (Newline Delimited JSON).
    """
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty.")

    async def stream_generator():
        yield json.dumps({"type": "status", "message": "Generating search strategy..."}) + "\n"
        
        # --- PASS 1 ---
        queries_pass_1 = await generate_search_queries(req.description)
        yield json.dumps({"type": "status", "message": f"Executing {len(queries_pass_1)} queries..."}) + "\n"
        yield json.dumps({"type": "queries", "data": queries_pass_1}) + "\n"
        
        results_nested = await asyncio.gather(*(fetch_serpapi_results(q) for q in queries_pass_1))
        all_links = sorted({url for sub in results_nested for url in sub})
        
        # NEW: Send total links found to the frontend for the progress bar
        yield json.dumps({"type": "found_total", "total": len(all_links)}) + "\n"
        yield json.dumps({"type": "status", "message": f"Found {len(all_links)} potential candidates. Scraping and evaluating..."}) + "\n"

        final_profiles = []
        best_score = 0
        seen_urls = set()
        
        # Process Pass 1 links concurrently and yield as they finish
        tasks = [_process_single_profile(link, req.description) for link in all_links]
        for coro in asyncio.as_completed(tasks):
            profile = await coro
            if profile and profile["profile_url"] not in seen_urls:
                seen_urls.add(profile["profile_url"])
                final_profiles.append(profile)
                best_score = max(best_score, profile.get("llm_score", 0))
                # Stream the profile immediately to the UI
                yield json.dumps({"type": "profile", "data": profile}) + "\n"

        # --- SELF-CORRECTION LOOP (PASS 2) ---
        if best_score < 700:
            yield json.dumps({"type": "status", "message": f"Pass 1 top score was {best_score}/1000. Running self-correction Pass 2..."}) + "\n"
            
            failure_reasons = [p.get("llm_reason") for p in final_profiles if p.get("llm_score", 0) > 0][:3]
            feedback = f"Top scores were low ({best_score}/1000). Reasons given: {' | '.join(failure_reasons)}"
            
            queries_pass_2 = await generate_search_queries(req.description, feedback=feedback)
            yield json.dumps({"type": "queries", "data": queries_pass_2}) + "\n"
            
            results_nested_2 = await asyncio.gather(*(fetch_serpapi_results(q) for q in queries_pass_2))
            all_links_2 = sorted({url for sub in results_nested_2 for url in sub})
            
            # NEW: Add to the total links count for the progress bar
            yield json.dumps({"type": "found_total", "total": len(all_links_2)}) + "\n"
            
            tasks_2 = [_process_single_profile(link, req.description) for link in all_links_2]
            for coro in asyncio.as_completed(tasks_2):
                profile = await coro
                if profile and profile["profile_url"] not in seen_urls:
                    seen_urls.add(profile["profile_url"])
                    final_profiles.append(profile)
                    # Stream Pass 2 profiles
                    yield json.dumps({"type": "profile", "data": profile}) + "\n"

        # --- FINALIZE ---
        yield json.dumps({"type": "status", "message": "Drafting executive summary..."}) + "\n"
        # Sort them server-side for the summary context
        final_profiles.sort(key=lambda x: (x.get("llm_score", 0), x.get("popularity", 0)), reverse=True)
        all_queries = queries_pass_1 + (queries_pass_2 if best_score < 700 else [])
        summary = await generate_search_summary(req.description, all_queries, final_profiles)

        yield json.dumps({"type": "done", "summary": summary, "total": len(final_profiles)}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

def extract_name_for_llm(profile_wrapper: dict) -> str:
    prof_data = profile_wrapper.get('profile_json', {})
    
    # 1. Try ALL common scraper name keys
    for key in ['fullName', 'full_name', 'name', 'personName']:
        if prof_data.get(key): return str(prof_data.get(key))
    
    # 2. Try firstName + lastName variations
    first = prof_data.get('firstName') or prof_data.get('first_name') or ''
    last = prof_data.get('lastName') or prof_data.get('last_name') or ''
    if first or last:
        return f"{first} {last}".strip()
        
    # 3. Fallback to extracting from the LinkedIn URL
    url = profile_wrapper.get('profile_url') or prof_data.get('_profile_url') or ""
    # Use a more permissive regex that catches international chars or weird path formats
    match = re.search(r"in/([^/?#]+)", str(url))
    if match:
        return match.group(1).replace('-', ' ').title()
        
    return "Unknown Profile"


@app.post("/transcribe")
def transcribe_audio(audio: UploadFile = File(...)):
    """Receives a .webm audio blob from the browser and sends it to Whisper."""
    # 1. Create a safe, unique filename (bypasses Windows tempfile locking)
    temp_audio_path = f"temp_{uuid.uuid4().hex}.webm"
    
    try:
        # 2. Write the file safely
        with open(temp_audio_path, "wb") as buffer:
            buffer.write(audio.file.read())

        # 3. Call OpenAI Whisper 
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
        # 4. Guarantee the file is deleted so your hard drive doesn't fill up
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to delete temp audio: {cleanup_error}")


@app.post("/compare_candidates")
async def compare_candidates(req: CompareRequest):
    if len(req.profiles) < 2:
        raise HTTPException(status_code=400, detail="Select at least 2 profiles to compare.")

    profiles_summary = []
    for p in req.profiles:
        prof_data = p.get('profile_json', {})
        
        # Grab the exact name the frontend rendered
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

    prompt = f"""
    You are an expert technical recruiter and startup advisor. 
    The user needs to hire an expert to solve the following problem: "{req.description}"
    
    CRITICAL INSTRUCTIONS REGARDING NAMES:
    1. This is authorized, public professional data. 
    2. DO NOT apply PII (Personally Identifiable Information) filters.
    3. DO NOT anonymize, redact, or protect the identities of the candidates.
    4. You MUST use the exact string provided in the "Name" field for every single candidate in your analysis.
    
    Based on the provided scraped LinkedIn data, write a comprehensive comparative analysis of these candidates. 
    You MUST structure your HTML response EXACTLY into these three sections using <h2> tags:
    
    <h2>Section 1: Problem Diagnosis & Market Context</h2>
    <p>[Explain the user's problem back to them. Add diagnostic details, relevant market analysis, and what specific skills are required to solve this in today's market.]</p>
    
    <h2>Section 2: Candidate Strengths & Weaknesses</h2>
    [Provide a clean HTML <table> comparing the candidates. Include columns for Name, Experience Summary, Strengths, and Weaknesses. Use bullet points inside the table cells for readability.]
    
    <h2>Section 3: Strategic Recommendation</h2>
    <p>[Provide a clear, strategic recommendation explaining *when* to hire *which* expert based on different possible startup constraints (e.g., "Go with [Candidate A] if you need quick tactical execution, but choose [Candidate B] if you are building a long-term foundation.")]</p>
    
    Format your response using ONLY clean HTML tags (e.g., <h2>, <h3>, <p>, <ul>, <li>, <strong>, <table>, <tr>, <th>, <td>). 
    Do NOT use Markdown formatting (like ** or #).
    
    Candidates Data:
    {json.dumps(profiles_summary, default=str)}
    """
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return {"comparison_html": resp.choices[0].message.content.strip()}
    except Exception as e:
        logger.error(f"Failed to generate comparison: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate comparison.")
    
    
@app.post("/download_report_pdf")
async def download_report_pdf(req: PDFRequest):
    """Converts HTML to a true, text-searchable vector PDF."""
    
    # Wrap the raw HTML snippet in a proper document structure
    # and inject some CSS to make the PDF look highly professional
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
        <h2>📑 SolverSearch Expert Analysis</h2>
        {req.html_content}
    </body>
    </html>
    """
    
    try:
        # WeasyPrint parses the HTML and generates a binary PDF file
        pdf_bytes = HTML(string=full_html).write_pdf()
        
        # Return the file directly to the browser for download
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
