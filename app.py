"""
Mindful Tourism App — Streamlit entry point.

Stage 1: User describes a trip → LLM returns 3-4 city cards (no RAG).
Stage 2: User selects a city → RAG + LLM generates a mindful travel guide.

Run:
    streamlit run app.py
"""
import os
import sys
import urllib.parse

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components

from llm.client import generate_mindful_tips, recommend_cities
from rag.cities import TARGET_CITIES          # lightweight import (no ML libs)
from rag.retriever import TravelRetriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mindful Tourism",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .stage-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .stage-active   { background: #2e7d32; color: #fff; }
    .stage-inactive { background: #9e9e9e; color: #fff; }
    .city-card-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 4px; }
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
    """
    Load the embedding model and open ChromaDB once per app session.
    `@st.cache_resource` ensures this is not recreated on every Streamlit rerun.
    """
    return TravelRetriever()

# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "stage":         1,
        "user_request":  "",
        "days":          3,
        "cities":        [],   # list[dict] from Stage 1 LLM
        "selected_city": None,
        "tips":          None, # dict from Stage 2 LLM
        "error":         None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ── Shared header ─────────────────────────────────────────────────────────────

st.title("🌍 Mindful Tourism")
st.caption("Discover destinations with intention. Travel slow, travel well.")

s1_cls = "stage-active"
s2_cls = "stage-active" if st.session_state.stage == 2 else "stage-inactive"
st.markdown(
    f'<span class="stage-badge {s1_cls}">① Discover Cities</span>'
    f'<span class="stage-badge {s2_cls}">② Mindful Guide</span>',
    unsafe_allow_html=True,
)
st.divider()

# ── Utility helpers ───────────────────────────────────────────────────────────

def _maps_embed_url(query: str) -> str:
    """Return a Google Maps embed URL for the given search query."""
    encoded = urllib.parse.quote_plus(query)
    return f"https://maps.google.com/maps?q={encoded}&output=embed&hl=en"


def _booking_url(hotel_name: str, city: str) -> str:
    """Return a Booking.com search URL for the given hotel and city."""
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
# STAGE 1 — City recommendation
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.stage == 1:

    st.subheader("Where do you want to go?")
    st.write("Describe your ideal trip and we'll suggest a few destinations.")

    with st.form("travel_form"):
        user_request = st.text_area(
            "Your travel idea",
            value=st.session_state.user_request,
            placeholder="e.g. From Barcelona, for swimming and relaxing",
            height=100,
        )
        days = st.slider(
            "Number of days",
            min_value=1, max_value=7,
            value=st.session_state.days,
            help="Used to tailor the day-by-day plan in Stage 2.",
        )
        submitted = st.form_submit_button("✈️  Find Destinations", type="primary")

    if submitted:
        if not user_request.strip():
            st.warning("Please describe your trip before searching.")
        else:
            st.session_state.user_request = user_request
            st.session_state.days         = days
            st.session_state.cities       = []
            st.session_state.error        = None

            with st.spinner("Consulting our travel AI …"):
                try:
                    cities = recommend_cities(
                        user_request,
                        candidate_cities=TARGET_CITIES,
                        days=days,
                    )
                    if not cities:
                        st.session_state.error = (
                            "The AI returned no suggestions. "
                            "Try rephrasing your request."
                        )
                    else:
                        st.session_state.cities = cities
                except Exception as exc:
                    st.session_state.error = str(exc)

    if st.session_state.error:
        st.error(st.session_state.error)

    if st.session_state.cities:
        st.subheader("Recommended Destinations")
        st.caption("Click **Select** on a city to get your personalised Mindful Guide.")

        cols = st.columns(len(st.session_state.cities), gap="medium")
        for i, city_data in enumerate(st.session_state.cities):
            city_name = city_data.get("city", "Unknown")
            with cols[i]:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="city-card-title">📍 {city_name}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"✈️ &nbsp;{city_data.get('flight_hours', '—')}")
                    st.markdown(f"💶 &nbsp;{city_data.get('price_estimate', '—')}")
                    st.write(city_data.get("reason", ""))
                    if st.button(
                        "Select this city →",
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

    if st.button("← Back to city selection"):
        _reset_to_stage1()
        st.rerun()

    st.subheader(f"Your Mindful Guide — {city}")
    st.caption(f"{days}-day trip · powered by WikiVoyage + AI")

    if st.session_state.tips is None and st.session_state.error is None:
        with st.spinner(f"Retrieving local knowledge for {city} …"):
            retriever = get_retriever()
            try:
                travel_context = retriever.retrieve(
                    city=city,
                    query=(
                        f"top attractions things to do etiquette tips "
                        f"pacing relaxation {city}"
                    ),
                )
            except Exception as exc:
                travel_context = ""
                st.warning(f"RAG retrieval issue (continuing without context): {exc}")

            try:
                hotel_context = retriever.retrieve(
                    city=city,
                    query=f"sleep accommodation hotel budget luxury hostel stay {city}",
                )
            except Exception as exc:
                hotel_context = ""

        with st.spinner("Crafting your personalised guide …"):
            try:
                tips = generate_mindful_tips(
                    city, days, travel_context,
                    user_request=st.session_state.user_request,
                    hotel_context=hotel_context,
                )
                if not tips:
                    st.session_state.error = (
                        "The AI returned an empty guide. Please try again."
                    )
                else:
                    st.session_state.tips = tips
            except Exception as exc:
                st.session_state.error = str(exc)

    if st.session_state.error:
        st.error(st.session_state.error)
        if st.button("Retry"):
            st.session_state.error = None
            st.rerun()

    if st.session_state.tips:
        tips = st.session_state.tips

        # ── Recommended Places ────────────────────────────────────────────────
        st.markdown("### 📍 Recommended Places")
        places = tips.get("recommended_places", [])
        if not places:
            st.info("No place recommendations returned by the AI.")
        else:
            for place in places:
                name = place.get("name", "—")
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.write(place.get("description", ""))
                    with st.expander("🗺️ View on Google Maps"):
                        components.html(
                            f'<iframe src="{_maps_embed_url(f"{name}, {city}")}" '
                            f'width="100%" height="300" style="border:0;" '
                            f'allowfullscreen loading="lazy"></iframe>',
                            height=310,
                        )

        st.divider()

        # ── Hotels ────────────────────────────────────────────────────────────
        st.markdown("### 🏨 Where to Stay")
        hotels = tips.get("hotels", [])
        if not hotels:
            st.info("No hotel data found in WikiVoyage for this city.")
        else:
            for hotel in hotels:
                hotel_name = hotel.get("name", "—")
                category   = hotel.get("category", "")
                badge      = {"budget": "🟢", "mid-range": "🟡", "luxury": "🔴"}.get(
                    category.lower(), "⚪"
                )
                with st.container(border=True):
                    col_info, col_btn = st.columns([3, 1])
                    with col_info:
                        st.markdown(f"{badge} **{hotel_name}** — *{category}*")
                        note = hotel.get("note", "")
                        if note:
                            st.caption(note)
                    with col_btn:
                        booking_url = _booking_url(hotel_name, city)
                        st.link_button("🔗 Booking.com", booking_url, use_container_width=True)

        st.divider()

        # ── Etiquette ─────────────────────────────────────────────────────────
        st.markdown("### 🙏 Local Etiquette")
        etiquette = tips.get("etiquette", [])
        if not etiquette:
            st.info("No etiquette tips returned by the AI.")
        else:
            for item in etiquette:
                st.markdown(f"✓ &nbsp;{item}")

        st.divider()

        # ── Pacing ────────────────────────────────────────────────────────────
        st.markdown("### 🌿 Mindful Pacing")
        pacing = tips.get("pacing_advice", "")
        if pacing:
            st.write(pacing)
        else:
            st.info("No pacing advice returned by the AI.")

        st.divider()
        st.caption(
            "🌱 *Mindful tourism means giving yourself permission to slow down, "
            "connect with local communities, and leave places better than you found them.*"
        )
