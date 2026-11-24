"""
utils/ui_styles.py
------------------
Central CSS injection for DataPilot.
Works in both Streamlit light AND dark mode.
All backgrounds use rgba() so they adapt to Streamlit's theme automatically.
No hardcoded #fff / #000 background or text colours.
"""

import streamlit as st

# ─── Indigo brand palette ──────────────────────────────────────────────────────
PRIMARY      = "#4F46E5"
PRIMARY_DARK = "#3730A3"
PRIMARY_GLOW = "rgba(79, 70, 229, 0.18)"
PRIMARY_RING = "rgba(79, 70, 229, 0.35)"
SUCCESS      = "#10B981"
WARNING      = "#F59E0B"
DANGER       = "#EF4444"

GLOBAL_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════
   GOOGLE FONT
═══════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════════════════════════════
   ROOT / BODY
═══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ═══════════════════════════════════════════════════════════════
   MAIN CONTAINER
═══════════════════════════════════════════════════════════════ */
.block-container {
    padding-top: 1.6rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR  — glass-card feel
═══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(79, 70, 229, 0.15) !important;
    backdrop-filter: blur(6px);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-emotion-cache-pkbazv {
    font-size: 0.85rem;
}
/* Sidebar nav links */
[data-testid="stSidebar"] a {
    border-radius: 8px !important;
    transition: background 0.2s;
}
[data-testid="stSidebar"] a:hover {
    background: rgba(79, 70, 229, 0.10) !important;
}

/* ═══════════════════════════════════════════════════════════════
   HEADER STRIP
═══════════════════════════════════════════════════════════════ */
[data-testid="stHeader"] {
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(79, 70, 229, 0.10);
}

/* ═══════════════════════════════════════════════════════════════
   TYPOGRAPHY HELPERS
═══════════════════════════════════════════════════════════════ */
h1 { font-weight: 800 !important; letter-spacing: -0.5px; }
h2 { font-weight: 700 !important; }
h3 { font-weight: 650 !important; }

/* ═══════════════════════════════════════════════════════════════
   METRIC CARDS  (st.metric)
═══════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: rgba(79, 70, 229, 0.06) !important;
    border: 1px solid rgba(79, 70, 229, 0.20) !important;
    border-radius: 16px !important;
    padding: 16px 20px !important;
    transition: transform 0.18s, box-shadow 0.18s;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.14);
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    opacity: 0.72 !important;
}
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #4F46E5 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
}

/* ═══════════════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    gap: 2px;
    border-bottom: 2px solid rgba(79, 70, 229, 0.15) !important;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 8px 18px !important;
    transition: background 0.18s, color 0.18s;
}
[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(79, 70, 229, 0.08) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: 3px solid #4F46E5 !important;
    color: #4F46E5 !important;
}

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════════ */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 20px !important;
    transition: transform 0.15s, box-shadow 0.15s;
    border: 1px solid rgba(79, 70, 229, 0.30) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.22) !important;
}
[data-testid="stButton"] > button[kind="primary"],
[data-testid="stButton"] > button.st-emotion-cache-primary {
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
    color: #fff !important;
    border: none !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.20) !important;
}

/* ═══════════════════════════════════════════════════════════════
   INPUTS / SELECTS / TEXT AREAS
═══════════════════════════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius: 10px !important;
    border: 1px solid rgba(79, 70, 229, 0.25) !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    outline: none !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border-radius: 14px !important;
    border: 2px dashed rgba(79, 70, 229, 0.30) !important;
    padding: 8px !important;
    transition: border-color 0.2s, background 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4F46E5 !important;
    background: rgba(79, 70, 229, 0.04) !important;
}

/* ═══════════════════════════════════════════════════════════════
   SLIDERS
═══════════════════════════════════════════════════════════════ */
[data-testid="stSlider"] [role="slider"] {
    background: #4F46E5 !important;
    border: 2px solid #fff !important;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.40) !important;
}
[data-testid="stSlider"] .st-emotion-cache-1dx1gwv,
[data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) {
    background: linear-gradient(90deg, #4F46E5, #6366F1) !important;
}

/* ═══════════════════════════════════════════════════════════════
   DATAFRAMES / TABLES
═══════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(79, 70, 229, 0.15) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ═══════════════════════════════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    border-radius: 14px !important;
    border: 1px solid rgba(79, 70, 229, 0.18) !important;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.10);
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 16px !important;
}

/* ═══════════════════════════════════════════════════════════════
   ALERT BOXES  (info, success, warning, error)
═══════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
}

/* ═══════════════════════════════════════════════════════════════
   MULTISELECT TAGS
═══════════════════════════════════════════════════════════════ */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background: rgba(79, 70, 229, 0.15) !important;
    color: #4F46E5 !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
}

/* ═══════════════════════════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #4F46E5, #8B5CF6) !important;
    border-radius: 999px !important;
}

/* ═══════════════════════════════════════════════════════════════
   PLOTLY CHART WRAPPER
═══════════════════════════════════════════════════════════════ */
[data-testid="stPlotlyChart"] {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid rgba(79, 70, 229, 0.12) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ═══════════════════════════════════════════════════════════════
   COLUMNS GAP HELPER
═══════════════════════════════════════════════════════════════ */
[data-testid="column"] {
    padding: 0 6px !important;
}

/* ═══════════════════════════════════════════════════════════════
   DIVIDER
═══════════════════════════════════════════════════════════════ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(79,70,229,0.30), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════
   CUSTOM UTILITY CLASSES  (used by all pages)
═══════════════════════════════════════════════════════════════ */

/* Page hero banner */
.dp-page-hero {
    background: linear-gradient(135deg, rgba(79,70,229,0.12) 0%, rgba(99,102,241,0.06) 100%);
    border: 1px solid rgba(79, 70, 229, 0.20);
    border-radius: 20px;
    padding: 24px 30px;
    margin-bottom: 20px;
}
.dp-page-hero h1 {
    margin: 0 0 4px 0;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.dp-page-hero p {
    margin: 0;
    opacity: 0.72;
    font-size: 0.92rem;
}
.dp-eyebrow {
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4F46E5;
    margin-bottom: 6px;
}

/* Stat / KPI card */
.dp-stat-card {
    background: rgba(79, 70, 229, 0.06);
    border: 1px solid rgba(79, 70, 229, 0.18);
    border-radius: 16px;
    padding: 16px 20px;
    transition: transform 0.18s, box-shadow 0.18s;
}
.dp-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(79,70,229,0.14);
}
.dp-stat-label {
    font-size: 0.70rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.65;
}
.dp-stat-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #4F46E5;
    line-height: 1.1;
    margin-top: 4px;
}
.dp-stat-sub {
    font-size: 0.75rem;
    opacity: 0.60;
    margin-top: 2px;
}

/* Section heading */
.dp-section-heading {
    font-size: 1.05rem;
    font-weight: 700;
    padding: 6px 0;
    border-bottom: 2px solid rgba(79,70,229,0.18);
    margin-bottom: 12px;
}

/* Insight / info pill */
.dp-pill {
    display: inline-block;
    background: rgba(79,70,229,0.10);
    color: #4F46E5;
    border: 1px solid rgba(79,70,229,0.25);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Badge variants */
.dp-badge-green {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    color: #10B981;
    border: 1px solid rgba(16,185,129,0.30);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.dp-badge-red {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    color: #EF4444;
    border: 1px solid rgba(239,68,68,0.30);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.dp-badge-amber {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    color: #F59E0B;
    border: 1px solid rgba(245,158,11,0.30);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Insight card (left-border style) */
.dp-insight-card {
    background: rgba(79,70,229,0.05);
    border-left: 4px solid #4F46E5;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.88rem;
    line-height: 1.5;
}

/* Chat message bubbles */
div[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    border: 1px solid rgba(79,70,229,0.14) !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
}

/* ═══════════════════════════════════════════════════════════════
   DARK MODE OVERRIDES
   Streamlit sets data-theme="dark" on <html> in dark mode.
═══════════════════════════════════════════════════════════════ */
[data-theme="dark"] [data-testid="stMetric"] {
    background: rgba(79, 70, 229, 0.10) !important;
    border-color: rgba(79, 70, 229, 0.28) !important;
}
[data-theme="dark"] [data-testid="stExpander"] {
    border-color: rgba(79, 70, 229, 0.25) !important;
}
[data-theme="dark"] .dp-stat-card {
    background: rgba(79, 70, 229, 0.10);
    border-color: rgba(79, 70, 229, 0.28);
}
[data-theme="dark"] [data-testid="stDataFrame"],
[data-theme="dark"] [data-testid="stTable"] {
    border-color: rgba(79, 70, 229, 0.22) !important;
}
[data-theme="dark"] [data-testid="stPlotlyChart"] {
    border-color: rgba(79, 70, 229, 0.20) !important;
}
[data-theme="dark"] .dp-page-hero {
    background: linear-gradient(135deg, rgba(79,70,229,0.18) 0%, rgba(99,102,241,0.09) 100%);
    border-color: rgba(79, 70, 229, 0.30);
}

/* ═══════════════════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79,70,229,0.30); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(79,70,229,0.55); }

/* ═══════════════════════════════════════════════════════════════
   SPINNER
═══════════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] > div {
    border-top-color: #4F46E5 !important;
}

/* ═══════════════════════════════════════════════════════════════
   TOOLTIP
═══════════════════════════════════════════════════════════════ */
[data-testid="stTooltipHoverTarget"] svg {
    color: rgba(79,70,229,0.60) !important;
}

/* ═══════════════════════════════════════════════════════════════
   CHECKBOX / RADIO
═══════════════════════════════════════════════════════════════ */
[data-testid="stCheckbox"] span[data-baseweb="checkbox"] div {
    border-color: #4F46E5 !important;
}

/* ═══════════════════════════════════════════════════════════════
   CAPTION / SMALL TEXT
═══════════════════════════════════════════════════════════════ */
[data-testid="stCaptionContainer"] {
    font-size: 0.78rem !important;
    opacity: 0.70;
}

</style>
"""


def inject_css() -> None:
    """Inject the global DataPilot CSS into the current Streamlit page."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
