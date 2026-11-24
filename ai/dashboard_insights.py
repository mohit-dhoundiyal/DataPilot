import hashlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from google import genai

MODEL_NAME = "gemini-3.5-flash-lite"


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


def gemini_is_configured():
    return bool(get_api_key())


def _safe_number(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return round(value, 3)
    except Exception:
        pass
    return None


def profile_dataset(df):
    """Create a compact, privacy-conscious profile for AI planning."""
    profile = {
        "rows": int(len(df)),
        "columns": [],
        "dataset_quality": {
            "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_cells": int(df.shape[0] * df.shape[1]),
        },
    }

    for col in df.columns:
        s = df[col]
        item = {
            "name": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2) if len(s) else 0,
            "unique": int(s.nunique(dropna=True)),
            "examples": [str(x)[:60] for x in s.dropna().astype(str).head(3).tolist()],
        }

        if pd.api.types.is_numeric_dtype(s):
            n = pd.to_numeric(s, errors="coerce").dropna()
            item["kind"] = "numeric"
            if not n.empty:
                item["statistics"] = {
                    "min": _safe_number(n.min()),
                    "max": _safe_number(n.max()),
                    "mean": _safe_number(n.mean()),
                    "median": _safe_number(n.median()),
                    "std": _safe_number(n.std()),
                }
        else:
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
            if len(s) and parsed.notna().mean() >= 0.75:
                item["kind"] = "date"
                valid = parsed.dropna()
                item["date_range"] = [str(valid.min()), str(valid.max())] if not valid.empty else []
            elif s.nunique(dropna=True) <= min(100, max(20, int(len(df) * 0.25))):
                item["kind"] = "categorical"
                item["top_values"] = {
                    str(k)[:80]: int(v)
                    for k, v in s.fillna("Missing").astype(str).value_counts().head(8).items()
                }
            else:
                item["kind"] = "text_or_high_cardinality"

        profile["columns"].append(item)

    return profile


def build_dashboard_candidates(df):
    """Calculate verifiable KPI, chart and insight candidates with pandas."""
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and 2 <= df[c].nunique(dropna=True) <= 30
    ]
    dates = []
    for c in df.columns:
        if c in numeric:
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        if len(df) and parsed.notna().mean() >= 0.75:
            dates.append(c)

    kpis = [
        {"id": "rows", "label": "Total Records", "value": int(len(df)), "format": "integer",
         "why": "Shows dataset size"},
        {"id": "columns", "label": "Total Fields", "value": int(df.shape[1]), "format": "integer",
         "why": "Shows dataset breadth"},
        {"id": "completeness", "label": "Data Completeness",
         "value": round(100 * (1 - df.isna().sum().sum() / max(df.size, 1)), 2),
         "format": "percent", "why": "Measures how complete the dataset is"},
        {"id": "duplicates", "label": "Duplicate Rows", "value": int(df.duplicated().sum()),
         "format": "integer", "why": "Highlights repeated records"},
    ]

    for c in numeric[:20]:
        values = pd.to_numeric(df[c], errors="coerce").dropna()
        if values.empty:
            continue
        safe = "".join(ch if ch.isalnum() else "_" for ch in str(c))[:35]
        kpis.extend([
            {"id": f"mean_{safe}", "label": f"Average {c}", "value": _safe_number(values.mean()),
             "format": "number", "why": f"Typical value of {c}", "column": str(c)},
            {"id": f"sum_{safe}", "label": f"Total {c}", "value": _safe_number(values.sum()),
             "format": "number", "why": f"Overall total of {c}", "column": str(c)},
        ])

    charts = []
    insights = []

    # Category distribution and group comparisons
    for cat in categorical[:12]:
        counts = df[cat].fillna("Missing").astype(str).value_counts().head(12)
        if len(counts):
            charts.append({
                "id": f"category_{cat}", "type": "category_bar",
                "x": str(cat), "title": f"Distribution by {cat}",
                "reason": f"Shows the most frequent groups in {cat}",
                "priority_hint": "Useful when category values are meaningfully different",
            })
            top_name, top_count = str(counts.index[0]), int(counts.iloc[0])
            share = round(top_count / max(len(df), 1) * 100, 1)
            insights.append({
                "id": f"topcat_{cat}", "text": f"{top_name} is the largest {cat} group with {top_count:,} records ({share}% of rows).",
                "kind": "category", "column": str(cat), "score": min(100, 45 + share),
            })

    # Numeric distributions and categorical comparisons
    for num in numeric[:15]:
        vals = pd.to_numeric(df[num], errors="coerce").dropna()
        if vals.empty:
            continue
        charts.append({
            "id": f"hist_{num}", "type": "histogram", "x": str(num),
            "title": f"Distribution of {num}",
            "reason": f"Shows spread, skew and possible unusual values in {num}",
        })
        if vals.std() > 0:
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
            if outliers:
                insights.append({
                    "id": f"outliers_{num}",
                    "text": f"{outliers:,} values in {num} fall outside the 1.5×IQR range; review them as potential outliers.",
                    "kind": "quality", "column": str(num),
                    "score": min(95, 50 + outliers / max(len(vals), 1) * 100),
                })

        for cat in categorical[:8]:
            grouped = df.groupby(cat, dropna=False)[num].mean().dropna().sort_values(ascending=False).head(10)
            if len(grouped) >= 2:
                charts.append({
                    "id": f"group_{num}_{cat}", "type": "group_bar",
                    "x": str(cat), "y": str(num),
                    "title": f"Average {num} by {cat}",
                    "reason": f"Compares {num} across {cat} groups",
                })

    # Date trends
    for date_col in dates[:5]:
        for num in numeric[:8]:
            temp = df[[date_col, num]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce", dayfirst=True)
            temp[num] = pd.to_numeric(temp[num], errors="coerce")
            temp = temp.dropna()
            if len(temp) < 3 or temp[date_col].nunique() < 2:
                continue
            charts.append({
                "id": f"trend_{date_col}_{num}", "type": "date_trend",
                "x": str(date_col), "y": str(num),
                "title": f"{num} trend over time",
                "reason": f"Reveals how {num} changes over {date_col}",
            })

    # Numeric relationships
    if len(numeric) >= 2:
        corr = df[numeric[:20]].corr(numeric_only=True)
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                value = corr.loc[a, b]
                if pd.notna(value) and abs(value) >= 0.45:
                    charts.append({
                        "id": f"scatter_{a}_{b}", "type": "scatter",
                        "x": str(a), "y": str(b),
                        "title": f"{a} vs {b}",
                        "reason": f"Potential relationship (correlation {value:.2f})",
                    })
                    insights.append({
                        "id": f"corr_{a}_{b}",
                        "text": f"{a} and {b} have a correlation of {value:.2f}. This is an association, not proof of causation.",
                        "kind": "relationship", "score": min(100, abs(float(value)) * 100),
                    })

    # Data quality findings
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    for col, pct in missing.head(8).items():
        if pct > 0:
            insights.append({
                "id": f"missing_{col}",
                "text": f"{col} has {pct:.1f}% missing values.",
                "kind": "quality", "column": str(col),
                "score": min(100, float(pct) * 1.5),
            })

    insights.sort(key=lambda x: x.get("score", 0), reverse=True)
    charts = charts[:100]
    return {
        "kpis": kpis[:50],
        "charts": charts,
        "insights": insights[:50],
        "available_columns": {
            "numeric": [str(x) for x in numeric],
            "categorical": [str(x) for x in categorical],
            "dates": [str(x) for x in dates],
        },
    }


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def _ask_gemini(profile_json, candidates_json, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""
You are the dashboard designer and senior data analyst inside DataPilot.
Your task is to choose the most valuable dashboard content for THIS dataset.
Use only the candidate IDs supplied. Never invent a column, metric, value, or chart.
Choose:
- 4 to 6 KPIs that tell the most useful story. Prefer meaningful business metrics
  over generic row/column counts when meaningful numeric metrics exist.
- 4 to 8 charts with varied purposes. Avoid redundant charts. Prefer charts that
  answer different questions and reveal distributions, comparisons, trends or relationships.
- 4 to 8 insight IDs, prioritizing surprising, actionable and material findings.
- A short dashboard title and subtitle that reflect the actual dataset.
- For each selected chart, a one-sentence question it helps answer.
Return ONLY valid JSON in this exact shape:
{{
 "dashboard_title": "string",
 "dashboard_subtitle": "string",
 "kpis": ["candidate_id"],
 "charts": [{{"id":"candidate_id","question":"string"}}],
 "insights": ["candidate_id"]
}}
Do not select charts whose required columns are absent. Keep choices diverse.
PROFILE:
{profile_json}
CANDIDATES:
{candidates_json}
"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={"temperature": 0.15, "max_output_tokens": 1800,
                "response_mime_type": "application/json"},
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty dashboard plan.")
    return json.loads(response.text)


def generate_dashboard_plan(profile, candidates):
    key = get_api_key()
    if not key:
        raise RuntimeError("Gemini API key is not configured.")
    profile_json = json.dumps(profile, ensure_ascii=False, default=str)
    candidates_json = json.dumps(candidates, ensure_ascii=False, default=str)
    return _ask_gemini(profile_json, candidates_json, key)


def validate_plan(plan, candidates):
    """Reject any AI-selected IDs that do not exist in our calculated candidates."""
    kpi_map = {x["id"]: x for x in candidates["kpis"]}
    chart_map = {x["id"]: x for x in candidates["charts"]}
    insight_map = {x["id"]: x for x in candidates["insights"]}

    kpis = [kpi_map[x] for x in plan.get("kpis", []) if x in kpi_map][:6]
    charts = []
    for item in plan.get("charts", []):
        chart_id = item.get("id") if isinstance(item, dict) else item
        if chart_id in chart_map and chart_id not in [c["id"] for c in charts]:
            chart = dict(chart_map[chart_id])
            chart["question"] = item.get("question", "") if isinstance(item, dict) else ""
            charts.append(chart)
    insights = [insight_map[x] for x in plan.get("insights", []) if x in insight_map][:8]

    if not kpis:
        kpis = candidates["kpis"][:4]
    if not charts:
        charts = candidates["charts"][:6]
    if not insights:
        insights = candidates["insights"][:6]

    return {
        "dashboard_title": str(plan.get("dashboard_title") or "Dataset Intelligence Dashboard")[:90],
        "dashboard_subtitle": str(plan.get("dashboard_subtitle") or "AI-selected insights from your data")[:180],
        "kpis": kpis,
        "charts": charts,
        "insights": insights,
    }
