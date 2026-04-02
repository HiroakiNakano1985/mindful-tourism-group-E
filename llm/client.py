"""
HuggingFace Inference API client — Qwen2.5-7B-Instruct.

Qwen2.5-Instruct natively supports the chat_completion (conversational) endpoint,
avoiding the provider routing issues that affect some other models.

Stage 1 — city recommendation: returns a JSON list of candidate cities.
Stage 2 — mindful tips:        returns a JSON object with spots, etiquette, pacing.
"""
import json
import os
import re

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

_HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
_MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"

# ── helpers ──────────────────────────────────────────────────────────────────

def _get_client() -> InferenceClient:
    """Return an InferenceClient for Qwen2.5-7B-Instruct."""
    return InferenceClient(model=_MODEL_ID, token=_HF_TOKEN)


def _find_json_structure(text: str, kind: str) -> str | None:
    """
    Extract the outermost JSON array or object from *text* using
    brace-counting rather than a regex, so nested structures are handled
    correctly (e.g. spots_plan containing arrays inside an object).

    Returns the raw JSON substring, or None if not found.
    """
    open_char  = "[" if kind == "array" else "{"
    close_char = "]" if kind == "array" else "}"
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == open_char:
            if start is None:
                start = i
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


def _extract_json(text: str, kind: str = "array"):
    """
    Pull the first JSON array or object out of raw LLM output.

    Steps
    -----
    1. Strip markdown code fences.
    2. Fix the common LLM mistake of duplicate opening braces ([ { { …).
    3. Use brace-counting extraction (handles arbitrary nesting).
    4. Attempt json.loads; return None on failure.
    """
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Fix common duplicate-brace artefact: [ { { → [ {
    text = re.sub(r"\[\s*\{\s*\{", "[{", text)
    text = re.sub(r"\}\s*\}\s*,", "},", text)
    text = re.sub(r"\}\s*\}\s*\]", "}]", text)

    raw = _find_json_structure(text, kind)
    if raw is None:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Last resort for arrays: extract each {...} object individually.
        # Works for city recommendation objects (all string values, no nesting).
        if kind == "array":
            objects = re.findall(r"\{[^{}]+\}", raw, re.DOTALL)
            try:
                return [json.loads(o) for o in objects] if objects else None
            except json.JSONDecodeError:
                pass
        return None


# ── Stage 1: city recommendation (no RAG) ────────────────────────────────────

def recommend_cities(
    user_request: str,
    candidate_cities: list[str] | None = None,
    days: int = 3,
) -> list[dict]:
    """
    Generate 3-4 destination city recommendations for the given free-form request.

    Parameters
    ----------
    user_request : str
        Free-form travel request from the user.
    candidate_cities : list[str] | None
        If provided, the LLM must choose only from this list.
        Typically the cities available in ChromaDB so Stage 2 RAG always hits.
    days : int
        Trip duration — passed to the LLM as context.

    Returns a list of dicts with keys:
        city           - "City Name, Country"
        reason         - why it suits the request
        flight_hours   - estimated flight duration from departure city
        price_estimate - rough round-trip budget in EUR
    Raises RuntimeError on API or parsing failure.
    """
    if candidate_cities:
        city_list = "\n".join(f"- {c}" for c in candidate_cities)
        city_constraint = (
            "You MUST choose ONLY from this list of available destinations:\n"
            f"{city_list}\n"
            "NEVER suggest any city outside this list."
        )
    else:
        city_constraint = "You may suggest any suitable destination."

    client = _get_client()
    try:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert travel advisor. "
                        "You ALWAYS respond with valid JSON only — "
                        "no prose, no markdown, no explanation. "
                        "CRITICAL RULE: if the user's request mentions a departure city "
                        "(e.g. 'from X', 'departing from X', 'leaving from X'), "
                        "that city is the ORIGIN and must NEVER appear as a recommendation."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Trip duration: {days} days\n"
                        f"Travel request: {user_request}\n\n"
                        f"{city_constraint}\n\n"
                        f"Choose 3-4 destinations that best match the request for a {days}-day trip. "
                        "Do NOT recommend the departure city mentioned in the request.\n"
                        + (
                            "This is a SHORT trip (1-3 days): prefer nearby destinations "
                            "(≤2 h flight) that can be comfortably explored in that time.\n"
                            if days <= 3 else
                            "All destinations in the list are reachable in the given days.\n"
                            "IMPORTANT: do NOT pick 3 cities from the same country or same region. "
                            "Spread recommendations across different countries to give the traveller genuine variety "
                            "(e.g. one Iberian, one Mediterranean island, one North Africa).\n"
                        )
                        + "\nReturn ONLY a JSON array — no extra text — with this exact structure.\n"
                        "CRITICAL: each object must start with exactly ONE opening brace {, never two.\n"
                        "[\n"
                        "  {\n"
                        '    "city": "City Name, Country",\n'
                        f'    "reason": "One sentence explaining why this city fits a {days}-day trip",\n'
                        '    "flight_hours": "X-Y hours from the departure city",\n'
                        '    "price_estimate": "€XXX–XXX approx return"\n'
                        "  }\n"
                        "]"
                    ),
                },
            ],
            max_tokens=700,
            temperature=0.7,
        )
        raw = response.choices[0].message.content or ""
        result = _extract_json(raw, kind="array")
        if not isinstance(result, list):
            raise ValueError(f"Expected JSON array, got: {raw[:200]}")
        return result
    except Exception as exc:
        raise RuntimeError(f"LLM city recommendation failed: {exc}") from exc


# ── Stage 2: mindful tips (with RAG context) ─────────────────────────────────

def generate_mindful_tips(
    city: str,
    days: int,
    context: str,
    user_request: str = "",
    hotel_context: str = "",
    etiquette_context: str = "",
    pacing_context: str = "",
) -> dict:
    """
    Generate mindful travel tips for *city* using retrieved RAG *context*.

    Parameters
    ----------
    city : str
        Selected destination city.
    days : int
        Trip duration in days.
    context : str
        RAG-retrieved text from ChromaDB (WikiVoyage, etc.).
    user_request : str
        Original free-form request from the user — used to tailor
        recommended places to their interests (e.g. beaches vs museums).

    Returns a dict with keys:
        recommended_places – list of {name, description} tailored to the request
        hotels             – list of {name, category, note} from WikiVoyage Sleep section
        etiquette          – list of short tip strings
        pacing_advice      – single paragraph string
    Raises RuntimeError on API or parsing failure.
    """
    interest_line = (
        f"The traveller's interests: {user_request}\n"
        "Tailor recommended_places to match these interests specifically.\n\n"
        if user_request else ""
    )

    client = _get_client()
    try:
        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a mindful travel advisor who cares about "
                        "sustainable, respectful tourism. "
                        "You ALWAYS respond with valid JSON only — "
                        "no prose, no markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Destination: {city}\n"
                        f"Trip length: {days} day(s)\n"
                        f"{interest_line}"
                        "--- Travel reference ---\n"
                        f"{context}\n"
                        "--- End travel reference ---\n\n"
                        "--- Accommodation reference ---\n"
                        f"{hotel_context if hotel_context else 'No accommodation data available.'}\n"
                        "--- End accommodation reference ---\n\n"
                        "--- Etiquette reference (tourist mistakes, scams, local norms) ---\n"
                        f"{etiquette_context if etiquette_context else 'No etiquette data available.'}\n"
                        "--- End etiquette reference ---\n\n"
                        "--- Pacing reference (timing, schedules, itinerary pitfalls) ---\n"
                        f"{pacing_context if pacing_context else 'No pacing data available.'}\n"
                        "--- End pacing reference ---\n\n"
                        "Using ALL references above, generate a mindful travel guide.\n"
                        "For hotels: extract real hotel names from the Accommodation reference. "
                        "If none are mentioned, use an empty array.\n"
                        "recommended_places must contain 5-8 places tailored to the traveller's interests.\n\n"
                        "CRITICAL INSTRUCTION FOR ETIQUETTE AND PACING:\n"
                        "You MUST extract specific facts from the Reference sections above.\n"
                        "DO NOT fall back on generic travel advice. If a tip could apply to any city, "
                        "it is too generic — rewrite it with a specific detail about " + city + ".\n"
                        "TONE: Write like a witty friend who's been there, not a guidebook. "
                        "Add humor — sarcasm, self-deprecation, funny warnings are encouraged. "
                        "Make the reader smile while learning something useful.\n\n"
                        f"etiquette: Give 5 tips SPECIFIC to {city}. Each tip must mention a "
                        "specific place, street, neighborhood, or custom unique to this city. "
                        "Pull from the Etiquette reference above. Examples of GOOD tips:\n"
                        f'  - "In {city}, restaurants add a coperto (cover charge) of €2-3 — this is normal, not a scam"\n'
                        f'  - "The waiters at Place X in {city} will ignore you if you wave — make eye contact instead"\n'
                        "Examples of BAD tips (too generic — NEVER write these):\n"
                        '  - "Be aware of pickpockets" / "Avoid tourist traps" / "Respect local customs"\n\n'
                        f"pacing_advice: Give 7-10 tips SPECIFIC to {city}. "
                        "This is a MINDFUL travel guide — focus on DEPTH of experience, not efficiency. "
                        "Each tip MUST mention a specific place, street, route, or time in this city. "
                        "Pull from the Pacing reference above. Cover ALL 5 categories:\n\n"
                        f"  1. TIME & SEASON: What's the magic hour in {city}? "
                        "Which cafe is locals-only before 8am? Which market has fresh fish only on Wednesday mornings?\n"
                        f"  2. LOCAL CONNECTION: Where to meet real locals in {city}? "
                        "A non-touristy bar, a sports match viewing spot, a food stall where the owner talks to you?\n"
                        f"  3. QUALITY OF MOVEMENT: A route that's slower but beautiful. "
                        "A bus ride with better views than the metro. A 20-min walk that beats a €10 taxi.\n"
                        f"  4. DOING NOTHING: Name a SPECIFIC place in {city} where doing nothing is the point. "
                        "A plaza with old men playing chess. A park kiosk selling €1 espresso. A hidden bench with a view. "
                        "Include a funny detail about why you'll accidentally spend 2 hours there.\n"
                        f"  5. LOCAL SCALE: How do locals in {city} think about time and distance? "
                        "When is lunch? Is 30min considered 'nearby'? What time is 'early' for dinner?\n\n"
                        "Examples of BAD tips (NEVER write these):\n"
                        '  - "Book in advance" / "Use public transport" / "Find a quiet bench"\n'
                        '  - Any tip that could apply to ANY city is too generic. Rewrite with {city}-specific detail.\n\n'
                        "Return ONLY a JSON object — no extra text — with this exact structure:\n"
                        "{\n"
                        '  "recommended_places": [\n'
                        '    {"name": "Place Name", "description": "Why visit and what to do, tailored to interests"}\n'
                        "  ],\n"
                        '  "hotels": [\n'
                        '    {"name": "Hotel Name", "category": "budget/mid-range/luxury", "note": "one-line highlight"}\n'
                        "  ],\n"
                        '  "etiquette": ["specific surprising tip 1", "specific surprising tip 2", "...", "...", "..."],\n'
                        '  "pacing_advice": ["timing tip", "transport tip", "reservation tip", "budget tip", "pacing tip"]\n'
                        "}"
                    ),
                },
            ],
            max_tokens=1400,
            temperature=0.7,
        )
        raw = response.choices[0].message.content or ""
        result = _extract_json(raw, kind="object")
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got: {raw[:200]}")
        return result
    except Exception as exc:
        raise RuntimeError(f"LLM tip generation failed: {exc}") from exc
