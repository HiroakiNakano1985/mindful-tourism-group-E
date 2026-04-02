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

import requests as http_requests
import streamlit as st
import streamlit.components.v1 as components

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
    /* Input fields in sidebar */
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
    /* Find Destinations button */
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
    /* Other buttons in sidebar */
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

    /* ── City cards ─────────────────────────────────── */
    .city-card img {
        border-radius: 10px 10px 0 0;
        width: 100%;
        height: 180px;
        object-fit: cover;
    }
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

    /* ── Hotel badge ────────────────────────────────── */
    .hotel-card {
        padding: 4px 0;
    }

    /* ── Data source tags ───────────────────────────── */
    .source-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin: 2px;
    }
    .tag-wiki   { background: #d8f3dc; color: #1b4332 !important; }
    .tag-reddit { background: #ffe0cc; color: #bf360c !important; }
    .tag-google { background: #dbeafe; color: #1e40af !important; }
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
    """
    Fetch a thumbnail image URL for *city_name* from the Wikipedia API.
    Returns a placeholder if none is found.
    """
    clean = city_name.split(",")[0].strip()
    params = {
        "action": "query",
        "titles": clean,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 400,
    }
    headers = {
        "User-Agent": "MindfulTourismApp/1.0 (educational prototype)"
    }
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

def _maps_embed_url(query: str) -> str:
    encoded = urllib.parse.quote_plus(query)
    return f"https://maps.google.com/maps?q={encoded}&output=embed&hl=en"

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


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Always visible
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🌍 Mindful Tourism")
    st.caption("Travel slow. Travel well.")
    st.divider()

    # ── Input form ────────────────────────────────────────────────────────────
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

    # ── Selected city indicator ───────────────────────────────────────────────
    if st.session_state.selected_city:
        st.divider()
        st.markdown("**Selected destination**")
        st.markdown(f"📍 **{st.session_state.selected_city}**")
        st.caption(f"{st.session_state.days}-day trip")
        if st.button("← Change city", use_container_width=True):
            _reset_to_stage1()
            st.rerun()

    # ── Data sources ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Data sources**")
    st.markdown("· WikiVoyage · Reddit · Google Places")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# ── Error banner ──────────────────────────────────────────────────────────────
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
        # Landing state
        st.markdown("# 🌍 Discover your next destination")
        st.markdown(
            "Describe your ideal trip in the sidebar and we'll suggest "
            "destinations tailored to your interests."
        )
    else:
        st.markdown("## Choose your destination")
        st.caption("Click **Select** to get your personalised Mindful Guide.")

        cols = st.columns(len(st.session_state.cities), gap="medium")
        for i, city_data in enumerate(st.session_state.cities):
            city_name = city_data.get("city", "Unknown")
            img_url   = get_city_image(city_name)

            with cols[i]:
                with st.container(border=True):
                    # City image
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

    # Collection shorthands for section-specific retrieval
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

    # ── Generate tips (cached in session_state) ───────────────────────────────
    if st.session_state.tips is None and st.session_state.error is None:
        with st.spinner(f"Retrieving local knowledge for {city} …"):
            retriever = get_retriever()

            try:
                travel_context = retriever.retrieve(
                    city=city,
                    query=(
                        f"top attractions things to do hidden gems "
                        f"local tips food {city}"
                    ),
                    collections=_ALL,
                )
            except Exception as exc:
                travel_context = ""
                st.warning(f"RAG retrieval issue: {exc}")

            try:
                hotel_context = retriever.retrieve(
                    city=city,
                    query=f"sleep accommodation hotel budget luxury hostel stay {city}",
                    collections=_HOTELS,
                )
            except Exception as exc:
                hotel_context = ""

            try:
                etiquette_context = retriever.retrieve(
                    city=city,
                    query=f"mistakes tourists make scams avoid tips warning rude {city}",
                    collections=_ALL,
                )
            except Exception as exc:
                etiquette_context = ""

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

        with st.spinner("Crafting your personalised guide …"):
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

    # ── Tips display ──────────────────────────────────────────────────────────
    if st.session_state.tips:
        tips = st.session_state.tips

        # ── Recommended Places (2 columns) ────────────────────────────────────
        st.markdown(
            '<div class="section-header">📍 Recommended Places</div>',
            unsafe_allow_html=True,
        )
        places = tips.get("recommended_places", [])
        if not places:
            st.info("No place recommendations returned by the AI.")
        else:
            for row_start in range(0, len(places), 2):
                row_places = places[row_start : row_start + 2]
                cols = st.columns(2, gap="medium")
                for j, place in enumerate(row_places):
                    name = place.get("name", "—")
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown(f"**{name}**")
                            st.write(place.get("description", ""))
                            with st.expander("🗺️ View on Map"):
                                components.html(
                                    f'<iframe src="{_maps_embed_url(f"{name}, {city}")}" '
                                    f'width="100%" height="250" style="border:0;border-radius:8px;" '
                                    f'allowfullscreen loading="lazy"></iframe>',
                                    height=260,
                                )

        st.divider()

        # ── Hotels (2 columns) ────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🏨 Where to Stay</div>',
            unsafe_allow_html=True,
        )
        hotels = tips.get("hotels", [])
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
                    with cols[j]:
                        with st.container(border=True):
                            st.markdown(f"{badge} **{hotel_name}**")
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

        # ── Etiquette ──────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🙏 Local Etiquette</div>',
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

        # ── Mindful Pacing (full width) ───────────────────────────────────
        st.markdown(
            '<div class="section-header">🌿 Mindful Pacing</div>',
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
