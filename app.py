"""
Mindful Tourism App — Streamlit entry point.

Stage 1: User describes a trip → LLM returns 3-4 city cards (no RAG).
Stage 2: User selects a city → RAG + LLM generates a mindful travel guide.

Run:
    streamlit run app.py --server.runOnSave true
"""
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import folium
import requests as http_requests
import streamlit as st
import streamlit.components.v1 as components
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium

from llm.client import generate_mindful_tips, recommend_cities
from rag.cities import (
    COLLECTION,
    GOOGLE_PLACES_COLLECTION,
    REDDIT_COLLECTION,
    TARGET_CITIES,
)
from rag.retriever import TravelRetriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mindful Tourism",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b4332 0%, #2d6a4f 100%);
    }
    section[data-testid="stSidebar"] *:not(.source-tag) {
        color: #d8f3dc !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stTextArea label {
        color: #b7e4c7 !important;
    }
    section[data-testid="stSidebar"] textarea {
        background: #1b4332 !important;
        color: #e8f5e9 !important;
        border: 1px solid #52b788 !important;
        border-radius: 8px !important;
        caret-color: #ffffff !important;
    }
    section[data-testid="stSidebar"] textarea::placeholder {
        color: #74c69d !important;
        opacity: 0.7 !important;
    }
    section[data-testid="stSidebar"] .stFormSubmitButton button {
        background: #f77f00 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    section[data-testid="stSidebar"] .stFormSubmitButton button:hover {
        background: #e36c00 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: #52b788 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #40916c !important;
    }

    /* ── City cards hover ───────────────────────────── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    /* ── City card content ──────────────────────────── */
    .city-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 8px 0 4px 0;
    }
    .city-card-meta {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 4px;
    }

    /* ── Section headers ────────────────────────────── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #52b788;
    }

    /* ── Section jump links ─────────────────────────── */
    .section-nav {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .section-nav a {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        background: #f0f2f6;
        color: #1b4332 !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 600;
        transition: background 0.2s;
    }
    .section-nav a:hover {
        background: #d8f3dc;
    }

    /* ── Sample prompt buttons ──────────────────────── */
    .prompt-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 1.5rem;
    }
    .prompt-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s;
    }
    .prompt-card:hover {
        background: #d8f3dc;
        border-color: #52b788;
        transform: translateY(-2px);
    }
    .prompt-card .emoji {
        font-size: 2rem;
        margin-bottom: 6px;
    }
    .prompt-card .label {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1b4332;
    }

    /* ── Loading steps ──────────────────────────────── */
    .loading-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        font-size: 0.9rem;
    }
    .loading-step .done { color: #2e7d32; }
    .loading-step .pending { color: #9e9e9e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── API token check ───────────────────────────────────────────────────────────

if not os.getenv("HUGGINGFACE_API_TOKEN"):
    st.error(
        "**HUGGINGFACE_API_TOKEN is not set.**  \n"
        "Copy `.env.example` to `.env` and add your token, then restart the app."
    )
    st.stop()

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_retriever() -> TravelRetriever:
    return TravelRetriever()


@st.cache_data(ttl=86400)
def get_city_image(city_name: str) -> str:
    clean = city_name.split(",")[0].strip()
    params = {
        "action": "query",
        "titles": clean,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 400,
    }
    headers = {"User-Agent": "MindfulTourismApp/1.0 (educational prototype)"}
    try:
        resp = http_requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params, headers=headers, timeout=8,
        )
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("thumbnail", {}).get("source", "")
    except Exception:
        return ""

# ── Sample prompts ────────────────────────────────────────────────────────────

SAMPLE_PROMPTS = [
    ("🏖️", "Beach & Relax",    "I want a slow quiet trip with beautiful beaches and clear water, from Barcelona"),
    ("🍕", "Street Food",       "I want to eat authentic local street food in hidden spots that only locals know about, from Barcelona"),
    ("🏛️", "History & Ruins",   "I want to visit off-the-beaten-path historical ruins and forgotten places, from Barcelona"),
    ("🌿", "Slow & Quiet",      "I want a slow quiet trip with no tourists, beautiful nature and local wine, from Barcelona"),
    ("🎵", "Nightlife",         "I want to explore underground music and nightlife scenes, from Barcelona"),
    ("👨‍👩‍👧", "Family Trip",       "Family trip with kids under 10, need safe beaches and fun activities, from Barcelona"),
]

# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "stage":         1,
        "user_request":  "",
        "days":          3,
        "cities":        [],
        "selected_city": None,
        "tips":          None,
        "error":         None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ── Utility helpers ───────────────────────────────────────────────────────────

@st.cache_data(ttl=86400)
def _geocode(place: str, city: str) -> tuple[float, float] | None:
    """
    Geocode a place name to (lat, lon).
    Tries Nominatim first (free), falls back to Google Places Text Search.
    """
    # Try Nominatim
    try:
        geolocator = Nominatim(user_agent="MindfulTourismApp/1.0")
        location = geolocator.geocode(f"{place}, {city}", timeout=5)
        if location:
            return (location.latitude, location.longitude)
    except Exception:
        pass

    # Fallback: Google Places Text Search
    google_key = os.getenv("GOOGLE_PLACES_API_KEY", "")
    if google_key:
        try:
            resp = http_requests.post(
                "https://places.googleapis.com/v1/places:searchText",
                json={"textQuery": f"{place}, {city}", "pageSize": 1},
                headers={
                    "Content-Type": "application/json",
                    "X-Goog-Api-Key": google_key,
                    "X-Goog-FieldMask": "places.location",
                },
                timeout=5,
            )
            places_data = resp.json().get("places", [])
            if places_data:
                loc = places_data[0].get("location", {})
                lat = loc.get("latitude")
                lng = loc.get("longitude")
                if lat and lng:
                    return (lat, lng)
        except Exception:
            pass

    return None


def _build_map(
    items: list[dict],
    city: str,
    name_key: str = "name",
    color_fn=None,
    icon_prefix: str = "info-sign",
) -> folium.Map | None:
    """
    Build a Folium map with markers for all items.
    color_fn: optional function(item) -> color string for the marker.
    Returns None if no items could be geocoded.
    """
    # First geocode the city itself for centering
    city_coords = _geocode(city.split(",")[0].strip(), "")
    if not city_coords:
        city_coords = (41.0, 2.0)  # fallback: Barcelona area

    m = folium.Map(location=city_coords, zoom_start=13, tiles="CartoDB positron")

    placed = 0
    for item in items:
        name = item.get(name_key, "")
        if not name:
            continue
        coords = _geocode(name, city)
        time.sleep(0.3)  # Nominatim rate limit: ~1 req/sec
        if not coords:
            continue

        color = color_fn(item) if color_fn else "blue"
        popup_text = name
        if "description" in item:
            popup_text += f"<br><small>{item['description'][:80]}...</small>"
        elif "note" in item:
            popup_text += f"<br><small>{item['note'][:80]}</small>"

        folium.Marker(
            location=coords,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon_prefix, prefix="glyphicon"),
        ).add_to(m)
        placed += 1

    return m if placed > 0 else None


def _hotel_color(hotel: dict) -> str:
    """Return marker color based on hotel category."""
    cat = hotel.get("category", "").lower()
    return {"budget": "green", "mid-range": "orange", "luxury": "red"}.get(cat, "blue")


def _booking_url(hotel_name: str, city: str) -> str:
    encoded = urllib.parse.quote_plus(f"{hotel_name} {city}")
    return f"https://www.booking.com/search.html?ss={encoded}"

def _reset_to_stage1():
    st.session_state.stage         = 1
    st.session_state.cities        = []
    st.session_state.selected_city = None
    st.session_state.tips          = None
    st.session_state.error         = None

def _select_city(city_name: str):
    st.session_state.selected_city = city_name
    st.session_state.tips          = None
    st.session_state.error         = None
    st.session_state.stage         = 2

def _use_sample_prompt(prompt_text: str):
    st.session_state.user_request = prompt_text


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Always visible
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 Mindful Tourism")
    st.caption("Travel slow. Travel well.")
    st.divider()

    st.markdown("**Plan your trip**")
    with st.form("travel_form"):
        user_request = st.text_area(
            "Your travel idea",
            value=st.session_state.user_request,
            placeholder="e.g. From Barcelona, for swimming and relaxing",
            height=140,
        )
        days = st.slider(
            "Number of days",
            min_value=1, max_value=7,
            value=st.session_state.days,
        )
        submitted = st.form_submit_button("✈️  Find Destinations", use_container_width=True)

    if submitted:
        if not user_request.strip():
            st.warning("Please describe your trip.")
        else:
            st.session_state.user_request = user_request
            st.session_state.days         = days
            st.session_state.cities       = []
            st.session_state.selected_city = None
            st.session_state.tips          = None
            st.session_state.error         = None
            st.session_state.stage         = 1

            with st.spinner("Consulting AI …"):
                try:
                    cities = recommend_cities(
                        user_request,
                        candidate_cities=TARGET_CITIES,
                        days=days,
                    )
                    if cities:
                        st.session_state.cities = cities
                    else:
                        st.session_state.error = "No suggestions. Try rephrasing."
                except Exception as exc:
                    st.session_state.error = str(exc)

    if st.session_state.selected_city:
        st.divider()
        st.markdown("**Selected destination**")
        st.markdown(f"📍 **{st.session_state.selected_city}**")
        st.caption(f"{st.session_state.days}-day trip")
        if st.button("← Change city", use_container_width=True):
            _reset_to_stage1()
            st.rerun()

    st.divider()
    st.markdown("**Data sources**")
    st.markdown("· WikiVoyage · Reddit · Google Places")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.error:
    st.error(st.session_state.error)
    if st.session_state.stage == 2:
        if st.button("Retry"):
            st.session_state.error = None
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — City recommendation
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.stage == 1:

    if not st.session_state.cities:
        # ── Landing page with sample prompts ──────────────────────────────────
        st.markdown("# 🌍 Discover your next destination")
        st.markdown(
            "Describe your ideal trip in the sidebar, or try one of these ideas:"
        )

        # Sample prompt buttons (2 rows of 3)
        cols = st.columns(3, gap="medium")
        for i, (emoji, label, prompt) in enumerate(SAMPLE_PROMPTS):
            with cols[i % 3]:
                if st.button(
                    f"{emoji} {label}",
                    key=f"sample_{i}",
                    use_container_width=True,
                ):
                    _use_sample_prompt(prompt)
                    st.rerun()

    else:
        # ── City cards ────────────────────────────────────────────────────────
        st.markdown("## Choose your destination")
        st.caption("Click **Select** to get your personalised Mindful Guide.")

        cols = st.columns(len(st.session_state.cities), gap="medium")
        for i, city_data in enumerate(st.session_state.cities):
            city_name = city_data.get("city", "Unknown")
            img_url   = get_city_image(city_name)

            with cols[i]:
                with st.container(border=True):
                    if img_url:
                        st.markdown(
                            f'<img src="{img_url}" style="border-radius:10px 10px 0 0;'
                            f'width:100%;height:180px;object-fit:cover;">',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="city-card-title">📍 {city_name}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="city-card-meta">'
                        f'✈️ {city_data.get("flight_hours", "—")} &nbsp;·&nbsp; '
                        f'💶 {city_data.get("price_estimate", "—")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(city_data.get("reason", ""))

                    if st.button(
                        "Select →",
                        key=f"select_{i}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _select_city(city_name)
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Mindful travel guide (RAG)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.stage == 2:

    city = st.session_state.selected_city
    days = st.session_state.days

    _ALL     = [COLLECTION, REDDIT_COLLECTION]
    _HOTELS  = [COLLECTION, GOOGLE_PLACES_COLLECTION]

    # ── Header with city image ────────────────────────────────────────────────
    img_url = get_city_image(city)
    if img_url:
        col_img, col_title = st.columns([1, 2])
        with col_img:
            st.image(img_url, use_container_width=True)
        with col_title:
            st.markdown(f"## {city}")
            st.caption(
                f"{days}-day trip · powered by WikiVoyage + Reddit + Google Places + AI"
            )
    else:
        st.markdown(f"## Your Mindful Guide — {city}")
        st.caption(f"{days}-day trip · powered by WikiVoyage + Reddit + Google Places + AI")

    # ── Section jump links ────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-nav">'
        '<a href="#your-mindful-map">🗺️ Map</a>'
        '<a href="#recommended-places">📍 Places</a>'
        '<a href="#where-to-stay">🏨 Hotels</a>'
        '<a href="#local-etiquette">🙏 Etiquette</a>'
        '<a href="#mindful-pacing">🌿 Pacing</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Generate tips with step-by-step progress ──────────────────────────────
    if st.session_state.tips is None and st.session_state.error is None:
        progress_container = st.empty()
        retriever = get_retriever()

        # Step 1: Travel context
        progress_container.markdown(
            '<div class="loading-step"><span class="pending">⏳</span> '
            'Searching local knowledge (places & food)...</div>',
            unsafe_allow_html=True,
        )
        try:
            travel_context = retriever.retrieve(
                city=city,
                query=f"top attractions things to do hidden gems local tips food {city}",
                collections=_ALL,
            )
        except Exception as exc:
            travel_context = ""

        # Step 2: Hotel context
        progress_container.markdown(
            '<div class="loading-step"><span class="done">✅</span> Local knowledge</div>'
            '<div class="loading-step"><span class="pending">⏳</span> '
            'Searching hotel reviews...</div>',
            unsafe_allow_html=True,
        )
        try:
            hotel_context = retriever.retrieve(
                city=city,
                query=f"sleep accommodation hotel budget luxury hostel stay {city}",
                collections=_HOTELS,
            )
        except Exception as exc:
            hotel_context = ""

        # Step 3: Etiquette context
        progress_container.markdown(
            '<div class="loading-step"><span class="done">✅</span> Local knowledge</div>'
            '<div class="loading-step"><span class="done">✅</span> Hotel reviews</div>'
            '<div class="loading-step"><span class="pending">⏳</span> '
            'Searching insider tips & warnings...</div>',
            unsafe_allow_html=True,
        )
        try:
            etiquette_context = retriever.retrieve(
                city=city,
                query=f"mistakes tourists make scams avoid tips warning rude {city}",
                collections=_ALL,
            )
        except Exception as exc:
            etiquette_context = ""

        # Step 4: Pacing context
        progress_container.markdown(
            '<div class="loading-step"><span class="done">✅</span> Local knowledge</div>'
            '<div class="loading-step"><span class="done">✅</span> Hotel reviews</div>'
            '<div class="loading-step"><span class="done">✅</span> Insider tips</div>'
            '<div class="loading-step"><span class="pending">⏳</span> '
            'Searching pacing & timing info...</div>',
            unsafe_allow_html=True,
        )
        try:
            pacing_context = retriever.retrieve(
                city=city,
                query=(
                    f"best time to visit early morning quiet spot locals only "
                    f"hidden park bench relax do nothing sit and watch "
                    f"slow walk scenic route bus vs walk "
                    f"closed sunday market schedule last train "
                    f"lunch time dinner time local pace {city}"
                ),
                collections=_ALL,
            )
        except Exception as exc:
            pacing_context = ""

        # Step 5: LLM generation
        progress_container.markdown(
            '<div class="loading-step"><span class="done">✅</span> Local knowledge</div>'
            '<div class="loading-step"><span class="done">✅</span> Hotel reviews</div>'
            '<div class="loading-step"><span class="done">✅</span> Insider tips</div>'
            '<div class="loading-step"><span class="done">✅</span> Pacing info</div>'
            '<div class="loading-step"><span class="pending">⏳</span> '
            'Crafting your personalised guide...</div>',
            unsafe_allow_html=True,
        )
        try:
            tips = generate_mindful_tips(
                city, days, travel_context,
                user_request=st.session_state.user_request,
                hotel_context=hotel_context,
                etiquette_context=etiquette_context,
                pacing_context=pacing_context,
            )
            if not tips:
                st.session_state.error = "The AI returned an empty guide. Please try again."
            else:
                st.session_state.tips = tips
        except Exception as exc:
            st.session_state.error = str(exc)

        progress_container.empty()

    # ── Tips display ──────────────────────────────────────────────────────────
    if st.session_state.tips:
        tips = st.session_state.tips

        # ── Combined Map (Places + Hotels) ────────────────────────────────────
        places = tips.get("recommended_places", [])
        hotels = tips.get("hotels", [])

        # Build one map with both places and hotels
        city_clean = city.split(",")[0].strip()
        city_coords = _geocode(city_clean, "")
        if not city_coords:
            city_coords = (41.0, 2.0)

        combined_map = folium.Map(
            location=city_coords, zoom_start=13, tiles="CartoDB positron"
        )
        marker_count = 0

        # Add places (numbered markers)
        place_num = 0
        for item in places:
            name = item.get("name", "")
            if not name:
                continue
            coords = _geocode(name, city)
            time.sleep(0.3)
            if not coords:
                continue
            place_num += 1
            desc = item.get("description", "")[:80]
            folium.Marker(
                location=coords,
                popup=folium.Popup(f"<b>#{place_num} {name}</b><br><small>{desc}...</small>", max_width=250),
                tooltip=f"#{place_num} {name}",
                icon=folium.DivIcon(
                    html=f'<div style="background:#2563eb;color:#fff;border-radius:50%;'
                         f'width:28px;height:28px;display:flex;align-items:center;'
                         f'justify-content:center;font-weight:700;font-size:14px;'
                         f'border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,0.3);">'
                         f'{place_num}</div>',
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
            ).add_to(combined_map)
            marker_count += 1
            item["_map_num"] = place_num  # store for card display

        # Add hotels (lettered markers)
        hotel_letters = "ABCDEFGHIJ"
        hotel_idx = 0
        for item in hotels:
            name = item.get("name", "")
            if not name:
                continue
            coords = _geocode(name, city)
            time.sleep(0.3)
            if not coords:
                continue
            letter = hotel_letters[hotel_idx % len(hotel_letters)]
            hotel_idx += 1
            note = item.get("note", "")[:80]
            cat = item.get("category", "")
            color_hex = {"budget": "#16a34a", "mid-range": "#ea580c", "luxury": "#dc2626"}.get(
                cat.lower(), "#6b7280"
            )
            folium.Marker(
                location=coords,
                popup=folium.Popup(f"<b>🏨{letter} {name}</b><br><small>{cat} — {note}</small>", max_width=250),
                tooltip=f"🏨{letter} {name}",
                icon=folium.DivIcon(
                    html=f'<div style="background:{color_hex};color:#fff;border-radius:50%;'
                         f'width:28px;height:28px;display:flex;align-items:center;'
                         f'justify-content:center;font-weight:700;font-size:14px;'
                         f'border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,0.3);">'
                         f'{letter}</div>',
                    icon_size=(28, 28),
                    icon_anchor=(14, 14),
                ),
            ).add_to(combined_map)
            marker_count += 1
            item["_map_letter"] = letter  # store for card display

        if marker_count > 0:
            st.markdown(
                '<div class="section-header">🗺️ Your Mindful Map</div>',
                unsafe_allow_html=True,
            )
            st.caption("🔵 1,2,3... = recommended places &nbsp;·&nbsp; 🟢A,B = budget hotel · 🟠A,B = mid-range · 🔴A,B = luxury")
            st_folium(combined_map, use_container_width=True, height=450, returned_objects=[])
            st.divider()

        # ── Recommended Places ────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header" id="recommended-places">📍 Recommended Places</div>',
            unsafe_allow_html=True,
        )
        if not places:
            st.info("No place recommendations returned by the AI.")
        else:
            for row_start in range(0, len(places), 2):
                row_places = places[row_start : row_start + 2]
                cols = st.columns(2, gap="medium")
                for j, place in enumerate(row_places):
                    name = place.get("name", "—")
                    num = place.get("_map_num", "")
                    label = f"**#{num} {name}**" if num else f"**{name}**"
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown(label)
                            st.write(place.get("description", ""))

        st.divider()

        # ── Hotels ────────────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header" id="where-to-stay">🏨 Where to Stay</div>',
            unsafe_allow_html=True,
        )
        if not hotels:
            st.info("No hotel data found for this city.")
        else:
            for row_start in range(0, len(hotels), 2):
                row_hotels = hotels[row_start : row_start + 2]
                cols = st.columns(2, gap="medium")
                for j, hotel in enumerate(row_hotels):
                    hotel_name = hotel.get("name", "—")
                    category   = hotel.get("category", "")
                    badge      = {"budget": "🟢", "mid-range": "🟡", "luxury": "🔴"}.get(
                        category.lower(), "⚪"
                    )
                    letter = hotel.get("_map_letter", "")
                    hotel_label = f"{badge} **{letter}. {hotel_name}**" if letter else f"{badge} **{hotel_name}**"
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown(hotel_label)
                            st.caption(f"{category}")
                            note = hotel.get("note", "")
                            if note:
                                st.write(note)
                            st.link_button(
                                "🔗 Booking.com",
                                _booking_url(hotel_name, city),
                                use_container_width=True,
                            )

        st.divider()

        # ── Etiquette ─────────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header" id="local-etiquette">🙏 Local Etiquette</div>',
            unsafe_allow_html=True,
        )
        etiquette = tips.get("etiquette", [])
        if not etiquette:
            st.info("No etiquette tips returned.")
        else:
            col_e1, col_e2 = st.columns(2, gap="medium")
            half = (len(etiquette) + 1) // 2
            with col_e1:
                for item in etiquette[:half]:
                    st.markdown(f"✓ &nbsp;{item}")
            with col_e2:
                for item in etiquette[half:]:
                    st.markdown(f"✓ &nbsp;{item}")

        st.divider()

        # ── Mindful Pacing ────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header" id="mindful-pacing">🌿 Mindful Pacing</div>',
            unsafe_allow_html=True,
        )
        pacing = tips.get("pacing_advice", "")
        if isinstance(pacing, list):
            col_p1, col_p2 = st.columns(2, gap="medium")
            half = (len(pacing) + 1) // 2
            with col_p1:
                for item in pacing[:half]:
                    with st.container(border=True):
                        st.markdown(f"💡 &nbsp;{item}")
            with col_p2:
                for item in pacing[half:]:
                    with st.container(border=True):
                        st.markdown(f"💡 &nbsp;{item}")
        elif pacing:
            st.write(pacing)
        else:
            st.info("No pacing advice returned.")

        st.divider()
        st.caption(
            "🌱 *Mindful tourism means giving yourself permission to slow down, "
            "connect with local communities, and leave places better than you found them.*"
        )
