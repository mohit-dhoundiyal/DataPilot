import os
import json
import hashlib
from datetime import date

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# DataPilot | AI Analyst
# Uses the latest cleaned dataframe when available.
# Gemini explains data; pandas performs all calculations.
# ---------------------------------------------------------

# Page configuration is managed by the main app.py file.


# -------------------- UI styling -------------------------
from utils.ui_styles import inject_css
inject_css()

# Extra AI Analyst specific styles (adaptive — rgba only)
st.markdown("""
<style>
.dp-hero {
    background: linear-gradient(135deg, rgba(79,70,229,0.12) 0%, rgba(99,102,241,0.06) 100%);
    border: 1px solid rgba(79,70,229,0.22);
    border-radius: 20px; padding: 26px 30px;
    margin-bottom: 22px;
}
.dp-eyebrow { color: #4F46E5; font-size: 12px; font-weight: 800; letter-spacing: 2px; text-transform: uppercase; }
.dp-title { font-size: 32px; font-weight: 800; margin: 5px 0 4px 0; }
.dp-subtitle { font-size: 14px; opacity: 0.72; margin: 0; }
.dp-card {
    background: rgba(79,70,229,0.06);
    border: 1px solid rgba(79,70,229,0.18);
    border-radius: 16px; padding: 17px 19px; min-height: 112px;
    transition: transform 0.18s, box-shadow 0.18s;
}
.dp-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(79,70,229,0.14); }
.dp-card-label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; opacity: 0.65; }
.dp-card-value { font-size: 27px; font-weight: 800; margin-top: 8px; color: #4F46E5; }
.dp-card-note { font-size: 11px; opacity: 0.60; margin-top: 2px; }
.dp-section { font-size: 19px; font-weight: 750; margin: 10px 0 12px 0; }
.dp-muted { font-size: 13px; opacity: 0.65; }
.dp-insight {
    background: rgba(79,70,229,0.07);
    border: 1px solid rgba(79,70,229,0.18);
    border-left: 4px solid #4F46E5;
    border-radius: 12px; padding: 14px 16px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Data selection --------------------
def get_active_dataframe():
    cleaned = st.session_state.get("cleaned_df")
    original = st.session_state.get("df")
    cleaned_flag = (
        st.session_state.get("cleaning_applied", False)
        or st.session_state.get("cleaning_round_number", 0) > 0
    )
    if isinstance(cleaned, pd.DataFrame) and cleaned_flag:
        return cleaned.copy(), "Latest cleaned dataset"
    if isinstance(original, pd.DataFrame):
        return original.copy(), "Original uploaded dataset"
    # Fallback for projects using another common session key.
    for key in ("data", "uploaded_df", "current_df"):
        candidate = st.session_state.get(key)
        if isinstance(candidate, pd.DataFrame):
            return candidate.copy(), "Active dataset"
    return None, None


df, data_source = get_active_dataframe()

def fmt_number(value):
    try:
        if pd.isna(value):
            return "—"
        value = float(value)
        if abs(value) >= 1_000_000:
            return f"{value/1_000_000:,.2f}M"
        if abs(value) >= 1_000:
            return f"{value:,.0f}"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)

def build_profile(data):
    """Compact, privacy-conscious profile. Does not send row-level records."""
    profile = {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_cells": int(data.isna().sum().sum()),
        "columns_detail": [],
        "numeric_summary": {},
        "categorical_summary": {},
        "date_columns": [],
        "correlations": [],
    }

    for col in data.columns:
        s = data[col]
        item = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique(dropna=True)),
        }
        non_null = s.dropna()
        if len(non_null):
            item["examples"] = [str(x)[:70] for x in non_null.astype(str).drop_duplicates().head(3).tolist()]
        profile["columns_detail"].append(item)

        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 1:
            profile["numeric_summary"][str(col)] = {
                "mean": round(float(s.mean()), 3) if s.notna().any() else None,
                "median": round(float(s.median()), 3) if s.notna().any() else None,
                "min": round(float(s.min()), 3) if s.notna().any() else None,
                "max": round(float(s.max()), 3) if s.notna().any() else None,
                "std": round(float(s.std()), 3) if s.notna().any() else None,
            }
        elif s.nunique(dropna=True) <= 30 and s.nunique(dropna=True) > 0:
            top = s.astype("string").fillna("(Missing)").value_counts().head(6)
            profile["categorical_summary"][str(col)] = {
                str(k)[:70]: int(v) for k, v in top.items()
            }

        if pd.api.types.is_datetime64_any_dtype(s):
            profile["date_columns"].append(str(col))
        elif s.dtype == "object" and len(non_null):
            parsed = pd.to_datetime(non_null.head(300), errors="coerce", dayfirst=True)
            if parsed.notna().mean() >= 0.8:
                profile["date_columns"].append(str(col))

    numeric = data.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        pairs = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1:]:
                val = corr.loc[c1, c2]
                if pd.notna(val):
                    pairs.append({"column_1": str(c1), "column_2": str(c2), "correlation": round(float(val), 3)})
        profile["correlations"] = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:8]
    return profile

def local_analysis(data):
    """Grounded calculations to accompany the AI prompt."""
    numeric = data.select_dtypes(include=np.number)
    output = {
        "numeric_columns": list(map(str, numeric.columns)),
        "category_columns": [str(c) for c in data.columns if data[c].nunique(dropna=True) <= 30],
        "top_correlations": [],
        "notable_numeric_ranges": {},
        "largest_categories": {},
        "quality": {
            "missing_by_column": {
                str(k): int(v) for k, v in data.isna().sum().sort_values(ascending=False).head(10).items() if v > 0
            },
            "duplicate_rows": int(data.duplicated().sum()),
        },
    }
    for col in numeric.columns:
        s = numeric[col].dropna()
        if len(s):
            output["notable_numeric_ranges"][str(col)] = {
                "min": round(float(s.min()), 3),
                "median": round(float(s.median()), 3),
                "max": round(float(s.max()), 3),
            }
    for col in data.columns:
        if col not in numeric.columns and 1 < data[col].nunique(dropna=True) <= 30:
            counts = data[col].astype("string").fillna("(Missing)").value_counts().head(5)
            output["largest_categories"][str(col)] = {str(k)[:60]: int(v) for k, v in counts.items()}
    if numeric.shape[1] >= 2:
        corr = numeric.corr(numeric_only=True)
        pairs = []
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[i + 1:]:
                v = corr.loc[c1, c2]
                if pd.notna(v):
                    pairs.append((str(c1), str(c2), float(v)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        output["top_correlations"] = [
            {"columns": [a, b], "correlation": round(v, 3)} for a, b, v in pairs[:10]
        ]
    return output

def get_api_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")

def ask_gemini(question, profile, analysis, history=None):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml."
        )
    from google import genai
    client = genai.Client(api_key=api_key)
    model_name = "gemini-3.5-flash-lite"
    prompt = f"""
You are DataPilot, a careful senior data analyst.
Answer the user's question using ONLY the supplied dataset profile and calculated analysis.
Never invent a value, claim causation from correlation, or pretend you inspected individual records.
If a requested calculation is not included, say so and explain what additional calculation is needed.
Give a useful answer in clear language. Use headings or concise bullets when helpful.
Mention the exact column names when discussing fields.
The active dataset source is: {data_source}

DATASET PROFILE:
{json.dumps(profile, ensure_ascii=False, default=str)}

CALCULATED ANALYSIS:
{json.dumps(analysis, ensure_ascii=False, default=str)}

RECENT CONVERSATION:
{json.dumps(history or [], ensure_ascii=False, default=str)}

USER QUESTION:
{question}
"""
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text or "Gemini returned an empty response."

def dataset_fingerprint(data):
    meta = f"{data.shape}|{list(data.columns)}|{list(map(str, data.dtypes))}|{int(data.isna().sum().sum())}"
    return hashlib.sha256(meta.encode("utf-8")).hexdigest()[:16]

# -------------------- Header -----------------------------
st.markdown(
    """
    <div class="dp-hero">
      <div class="dp-eyebrow">DATAPILOT · INTELLIGENCE WORKSPACE</div>
      <div class="dp-title">AI Data Analyst <span style="color:#a5b4fc">✦</span></div>
      <p class="dp-subtitle">Ask questions, uncover patterns, and turn your dataset into clear business insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if df is None or df.empty:
    st.warning("No dataset found. Upload a CSV or Excel file from the Data Profile page first.")
    st.stop()

fingerprint = dataset_fingerprint(df)
if st.session_state.get("ai_analyst_fingerprint") != fingerprint:
    st.session_state["ai_analyst_fingerprint"] = fingerprint
    st.session_state["ai_analyst_messages"] = []
    st.session_state["ai_analyst_profile"] = None
    st.session_state["ai_analyst_analysis"] = None

profile = build_profile(df)
analysis = local_analysis(df)

# -------------------- Dataset overview ------------------
source_col, status_col = st.columns([3, 1])
with source_col:
    st.markdown(f"<div class='dp-muted'>DATA SOURCE · <b style='color:#c7d2fe'>{data_source}</b></div>", unsafe_allow_html=True)
with status_col:
    st.markdown("<div style='text-align:right;color:#86efac;font-size:12px;font-weight:700'>● DATA READY</div>", unsafe_allow_html=True)

complete_pct = 100 * (1 - (df.isna().sum().sum() / max(df.size, 1)))
k1, k2, k3, k4 = st.columns(4)
cards = [
    ("TOTAL RECORDS", fmt_number(len(df)), "Rows available for analysis"),
    ("DATA FIELDS", fmt_number(df.shape[1]), "Columns detected"),
    ("COMPLETENESS", f"{complete_pct:.1f}%", "Non-missing cells"),
    ("DUPLICATE ROWS", fmt_number(df.duplicated().sum()), "Exact duplicate records"),
]
for col, (label, value, note) in zip([k1, k2, k3, k4], cards):
    with col:
        st.markdown(
            f"<div class='dp-card'><div class='dp-card-label'>{label}</div><div class='dp-card-value'>{value}</div><div class='dp-card-note'>{note}</div></div>",
            unsafe_allow_html=True,
        )

st.write("")
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown("<div class='dp-section'>✦ Ask your data anything</div>", unsafe_allow_html=True)
    st.caption("Gemini interprets your question; pandas supplies the dataset profile and calculated facts.")
    suggestions = [
        "What are the most important things I should know about this dataset?",
        "Which columns have data-quality problems?",
        "What relationships or correlations stand out?",
        "Which segments or categories deserve attention?",
    ]
    prompt_cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with prompt_cols[i % 2]:
            if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                st.session_state["ai_analyst_pending_prompt"] = suggestion

    for message in st.session_state.get("ai_analyst_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about your dataset...")
    if st.session_state.get("ai_analyst_pending_prompt"):
        user_question = st.session_state.pop("ai_analyst_pending_prompt")

    if user_question:
        st.session_state.setdefault("ai_analyst_messages", [])
        st.session_state["ai_analyst_messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing your dataset..."):
                try:
                    recent = st.session_state["ai_analyst_messages"][-7:-1]
                    answer = ask_gemini(user_question, profile, analysis, recent)
                    st.markdown(answer)
                    st.session_state["ai_analyst_messages"].append({"role": "assistant", "content": answer})
                except Exception as exc:
                    error_text = f"**Could not complete the AI analysis.** {exc}"
                    st.error(error_text)
                    st.session_state["ai_analyst_messages"].append({"role": "assistant", "content": error_text})

    if st.button("Clear conversation", key="clear_ai_chat"):
        st.session_state["ai_analyst_messages"] = []
        st.rerun()

with right:
    st.markdown("<div class='dp-section'>◈ Dataset intelligence</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="dp-insight"><strong>Dataset shape</strong><br>
        <span class="dp-muted">{len(df):,} records across {df.shape[1]} columns.</span></div>
        <div class="dp-insight"><strong>Data completeness</strong><br>
        <span class="dp-muted">{complete_pct:.1f}% of cells contain values.</span></div>
        <div class="dp-insight"><strong>Numeric fields</strong><br>
        <span class="dp-muted">{len(profile['numeric_summary'])} numeric columns detected for statistical analysis.</span></div>
        <div class="dp-insight"><strong>Potential date fields</strong><br>
        <span class="dp-muted">{', '.join(profile['date_columns']) if profile['date_columns'] else 'No clear date columns detected.'}</span></div>
        """,
        unsafe_allow_html=True,
    )
    if analysis["top_correlations"]:
        strongest = analysis["top_correlations"][0]
        st.markdown(
            f"<div class='dp-insight'><strong>Strongest numeric relationship</strong><br><span class='dp-muted'>{strongest['columns'][0]} ↔ {strongest['columns'][1]} · correlation {strongest['correlation']}</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='dp-insight'><strong>Relationships</strong><br><span class='dp-muted'>Need at least two numeric columns to calculate correlations.</span></div>", unsafe_allow_html=True)



with st.expander("Privacy and AI usage"):
    st.write(
        "Gemini receives column names, data types, aggregate statistics, category counts, "
        "missing-value summaries and correlations. The full dataset and individual rows are "
        "not sent by this page. Dashboard calculations and previews run locally with pandas."
    )
