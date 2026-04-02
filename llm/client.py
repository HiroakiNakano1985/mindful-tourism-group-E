"""
Google Gemini API client — gemini-3-flash-preview.

Stage 1 — city recommendation (1 LLM call).
Stage 2 — 4 parallel LLM calls: places, hotels, etiquette, pacing.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

_API_KEY  = os.getenv("GEMINI_API_KEY", "")
_MODEL_ID = "gemini-2.5-flash-lite"

_client = genai.Client(api_key=_API_KEY)

# ── helpers ──────────────────────────────────────────────────────────────────

def _find_json_structure(text: str, kind: str) -> str | None:
    """Extract the outermost JSON array or object using brace-counting."""
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
    """Pull the first JSON array or object out of raw LLM output."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    raw = _find_json_structure(text, kind)

    # If brace-counting found a complete structure, try to parse it
    if raw is not None:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Fallback: extract individual {...} objects (works for truncated arrays too)
    if kind == "array":
        objects = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
        if objects:
            parsed = []
            for o in objects:
                try:
                    parsed.append(json.loads(o))
                except json.JSONDecodeError:
                    continue
            return parsed if parsed else None

    return None


def _parse_list_response(raw: str, label: str = "") -> list:
    """
    Parse a JSON response that should be a list.
    Handles both plain arrays and object wrappers like {"key": [...]}.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Extract the first list value from the dict
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    # Fallback to extraction
    result = _extract_json(raw, kind="array")
    if isinstance(result, list):
        return result
    print(f"[warn] {label}: could not parse response: {raw[:200]}")
    return []


def _generate(system: str, user: str, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """Send a prompt to Gemini and return the response text."""
    response = _client.models.generate_content(
        model=_MODEL_ID,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


# ── Shared system instruction fragments ───────────────────────────────────────

_SYSTEM_BASE = (
    "You are a mindful travel advisor who writes like a witty friend, not a guidebook. "
    "Add humor — sarcasm, self-deprecation, funny warnings are encouraged. "
    "You ALWAYS respond with valid JSON only — no prose, no markdown."
)

_SPECIFICITY_RULE = (
    "CRITICAL: every tip must be SPECIFIC to {city}. "
    "If a tip could apply to ANY city, it is too generic — rewrite it with "
    "a specific place, street, or custom unique to {city}. "
    "DO NOT fall back on generic travel advice."
)


# ── Stage 1: city recommendation ─────────────────────────────────────────────

def recommend_cities(
    user_request: str,
    candidate_cities: list[str] | None = None,
    days: int = 3,
) -> list[dict]:
    """Generate 3-4 destination city recommendations."""
    if candidate_cities:
        city_list = "\n".join(f"- {c}" for c in candidate_cities)
        city_constraint = (
            "You MUST choose ONLY from this list of available destinations:\n"
            f"{city_list}\n"
            "NEVER suggest any city outside this list."
        )
    else:
        city_constraint = "You may suggest any suitable destination."

    system_msg = (
        "You are a travel advisor. Respond with valid JSON only. No markdown, no prose."
    )

    short_trip = "Prefer nearby destinations (≤2h flight)." if days <= 3 else ""

    user_msg = (
        f"Travel request: {user_request}\n"
        f"Trip: {days} days\n\n"
        f"{city_constraint}\n\n"
        "RULES:\n"
        "1. Return EXACTLY 4 cities. Not 1, not 2, not 3. Four.\n"
        "2. Never recommend the departure city mentioned in the request.\n"
        "3. Max 1 city per country.\n"
        f"4. {short_trip}\n\n"
        "TAGS RULE: each city must have exactly 3 tags (max 4 words each). "
        "Each tag MUST start with an emoji that represents the vibe. "
        "Tags describe the VIBE, not venue names. "
        "GOOD: '🍷 Wine marathon vineyards', '🎵 Fado bars at 2am', '🌅 Hilltop sunset kiosks' "
        "BAD: 'Village Underground LX', 'Beautiful city', 'Great nightlife'\n\n"
        "Return a JSON array of 4 objects:\n"
        "[{\"city\": \"Name, Country\", \"tags\": [\"tag1\", \"tag2\", \"tag3\"], "
        "\"flight_hours\": \"X-Y hours\", \"price_estimate\": \"€XXX-XXX\"}]"
    )

    try:
        for attempt in range(2):
            temp = 0.3 if attempt == 0 else 0.5
            raw = _generate(system_msg, user_msg, max_tokens=4000, temperature=temp)
            result = _parse_list_response(raw, "cities")
            if len(result) >= 3:
                return result
        # Return whatever we got on last attempt
        if result:
            return result
        raise ValueError(f"Expected JSON array, got: {raw[:200]}")
    except Exception as exc:
        raise RuntimeError(f"LLM city recommendation failed: {exc}") from exc


# ── Stage 2: four parallel LLM calls ─────────────────────────────────────────

def _generate_places(
    city: str, days: int, context: str, user_request: str,
    tags: list[str] | None = None,
) -> list:
    """Generate recommended places with mindful moments."""
    interest = f"The traveller's interests: {user_request}\n" if user_request else ""
    user_msg = (
        f"Destination: {city}, Trip: {days} day(s)\n"
        f"{interest}\n"
        f"--- Reference ---\n{context}\n--- End ---\n\n"
        f"Recommend 5-8 places in {city} tailored to the traveller's interests.\n"
        + (
            "The traveller chose this city because of these vibes: "
            + ", ".join(tags) + ". "
            "You MUST include at least one place that directly matches each of these vibes.\n"
            if tags else ""
        )
        + _SPECIFICITY_RULE.format(city=city) + "\n"
        "For each place, include a 'mindful_moment': a short, specific suggestion "
        "for how to EXPERIENCE this place (not just see it). "
        "Examples: 'Sit in the back courtyard for 10 minutes without your phone — "
        "notice the art on the ceiling that everyone misses.' or "
        "'Walk through without buying anything first. Come back to the stall "
        "with the best smell.'\n\n"
        "Return ONLY a JSON array:\n"
        '[{"name": "Place Name", "description": "Why visit (2-3 sentences)", '
        '"mindful_moment": "How to experience this place, not just see it"}]'
    )
    raw = _generate(_SYSTEM_BASE, user_msg, max_tokens=8000)
    return _parse_list_response(raw, "places")


def _generate_hotels(city: str, hotel_context: str) -> list:
    """Extract hotels from Google Places reviews."""
    user_msg = (
        f"City: {city}\n\n"
        f"--- Hotel reviews ---\n{hotel_context}\n--- End ---\n\n"
        "The reviews above contain lines like '[Google Review of HOTEL NAME, rating:X/5]'.\n"
        "Extract 3-5 distinct hotels from these review headers.\n"
        "For each hotel, determine the category (budget/mid-range/luxury) from review content,\n"
        "and write a one-line witty highlight based on what reviewers said.\n\n"
        "Return ONLY a JSON array:\n"
        '[{"name": "Hotel Name", "category": "budget/mid-range/luxury", "note": "one-line highlight from reviews"}]'
    )
    raw = _generate(_SYSTEM_BASE, user_msg, max_tokens=8000)
    return _parse_list_response(raw, "hotels")


def _generate_etiquette(city: str, etiquette_context: str) -> list:
    """Generate local etiquette tips."""
    user_msg = (
        f"City: {city}\n\n"
        f"--- Reference ---\n{etiquette_context}\n--- End ---\n\n"
        f"Give 5 etiquette tips SPECIFIC to {city}. Each tip must mention a "
        "specific place, street, neighborhood, or custom unique to this city.\n"
        + _SPECIFICITY_RULE.format(city=city) + "\n"
        "Examples of GOOD tips:\n"
        f'  - "In {city}, restaurants add a coperto of €2-3 — normal, not a scam"\n'
        f'  - "The waiters at Place X in {city} will ignore you if you wave — make eye contact"\n'
        "Examples of BAD tips (NEVER write):\n"
        '  - "Be aware of pickpockets" / "Respect local customs"\n\n'
        "Return ONLY a JSON array of 5 strings:\n"
        '["tip 1", "tip 2", "tip 3", "tip 4", "tip 5"]'
    )
    raw = _generate(_SYSTEM_BASE, user_msg, max_tokens=8000)
    return _parse_list_response(raw, "etiquette")


def _generate_pacing(city: str, days: int, pacing_context: str) -> list:
    """Generate mindful pacing tips."""
    user_msg = (
        f"City: {city}, Trip: {days} day(s)\n\n"
        f"--- Reference ---\n{pacing_context}\n--- End ---\n\n"
        f"Give exactly 8 mindful pacing tips SPECIFIC to {city}. "
        "Focus on DEPTH of experience, not efficiency.\n"
        + _SPECIFICITY_RULE.format(city=city) + "\n"
        "Cover these 6 categories (at least 1 tip each):\n"
        f"  1. TIME & SEASON: magic hour in {city}? locals-only cafe before 8am?\n"
        f"  2. LOCAL CONNECTION: where to meet real locals? non-touristy bar?\n"
        f"  3. QUALITY OF MOVEMENT: slower but beautiful route? walk that beats a taxi?\n"
        f"  4. DOING NOTHING: specific place where doing nothing is the point? "
        "Include a funny detail about why you'll spend 2 hours there.\n"
        f"  5. LOCAL SCALE: when is lunch in {city}? what's 'nearby'?\n"
        f"  6. SUSTAINABILITY: a specific eco-friendly tip for {city} "
        "(e.g. 'tap water is drinkable', 'tram X has the same view as the €40 boat tour')\n\n"
        "Examples of BAD tips (NEVER write):\n"
        '  - "Book in advance" / "Use public transport" / "Find a quiet bench"\n\n'
        "Return ONLY a JSON array of exactly 8 strings:\n"
        '["tip 1", "tip 2", ...]'
    )
    raw = _generate(_SYSTEM_BASE, user_msg, max_tokens=8000)
    return _parse_list_response(raw, "pacing")


def generate_mindful_tips(
    city: str,
    days: int,
    context: str,
    user_request: str = "",
    hotel_context: str = "",
    etiquette_context: str = "",
    pacing_context: str = "",
    tags: list[str] | None = None,
) -> dict:
    """
    Generate all four sections in parallel using ThreadPoolExecutor.
    Returns a combined dict with all sections.
    """
    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _generate_places, city, days, context, user_request, tags,
            ): "recommended_places",
            executor.submit(_generate_hotels, city, hotel_context): "hotels",
            executor.submit(_generate_etiquette, city, etiquette_context): "etiquette",
            executor.submit(
                _generate_pacing, city, days, pacing_context,
            ): "pacing_advice",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                print(f"[warn] {key}: {exc}")
                results[key] = []

    if not any(results.values()):
        return {}

    return results
