"""
Chart planning for the DataPilot dashboard.  (No Streamlit imports -> easy to test.)

Works on ANY tabular dataset. Nothing here refers to specific column names;
decisions come from column TYPES, cardinality, value patterns and keywords.

Two planners produce the SAME plan format:
  * plan_rules()  - instant, free, offline (default)
  * Gemini        - build_prompt() -> call_gemini() -> validate_plan()
Gemini only *chooses* columns/chart types from a fixed menu; every number is
computed locally with pandas and every Gemini answer is validated first.
"""
import calendar
import json
import re

import numpy as np
import pandas as pd

try:  # reuse the identifier rules from the cleaning page
    from engine.cleaner import is_identifier_column as _cleaner_is_id
except Exception:  # pragma: no cover
    _cleaner_is_id = None

CHART_TYPES = {
    "top_bar": "ranked horizontal bars: top N of a dimension by a metric",
    "column_rank": "ranked vertical columns: top N of a dimension (use for a 2nd ranking)",
    "trend": "metric over time (needs a date or period column; optional metric2 as 2nd line)",
    "donut": "share of a metric across a dimension with 3-8 groups",
    "stacked": "stacked bars of a dimension (stack) over time, or over dimension2",
    "seasonality": "metric by month of year (needs a real date column)",
    "heatmap": "average of a metric across dimension x dimension2",
    "scatter": "metric vs metric2, optional dimension as colour",
    "distribution": "histogram + box plot of a metric",
    "box": "metric spread per group of a dimension",
    "drivers": "ranked factors influencing the focus metric (computed by the app)",
}
AGGS = {"sum", "mean", "median", "count"}
UNKNOWN = "Unknown"
_PREF = ("amount", "revenue", "sales", "profit", "enroll", "price", "cost", "spend",
         "income", "total", "salary", "complet", "issued", "quantity", "units", "fare", "value")
_MEAN_WORDS = ("rate", "rating", "score", "percent", "pct", "ratio", "avg", "average", "age",
               "price", "margin", "satisfaction", "temperature", "duration", "fare")
_METRICISH = re.compile(r"rating|score|amount|price|revenue|sales|count|total|qty|quantity|age|"
                        r"cost|profit|salary|income|spend|units|number|num|fare|value|rate")
_DATE_HINT = re.compile(r"date|time|timestamp|created|updated|dob|birth|joined|joining|signup|"
                        r"published|issued|day$")
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS["sept"] = 9
_PERIOD_ORDER = ("month", "quarter", "fy", "year", "month_num")


# ---------------------------------------------------------- column roles
def _nat(x):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(x))]


def detect_period(s: pd.Series, name: str):
    """Month names, quarters, financial years, years -> 'period' columns (not metrics)."""
    if s.dropna().empty or pd.api.types.is_datetime64_any_dtype(s):
        return None
    lname = name.lower().strip()
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    vals = s.dropna()
    sv = vals.astype(str).str.strip().str.lower()
    nun = sv.nunique()
    if not numeric and nun <= 12 and sv.isin(_MONTHS).mean() >= 0.9:
        return "month"
    if nun <= 30 and (re.search(r"(^|[\s_\-])(fy|fiscal[\s_\-]*year|financial[\s_\-]*year)($|[\s_\-])", lname)
                      or (not numeric and sv.str.match(r"^(fy|fiscal)[\s\-_]*\d{2,4}").mean() >= 0.9)):
        return "fy"
    if nun <= 8 and re.search(r"quarter|(^|[\s_\-])qtr($|[\s_\-])", lname):
        return "quarter"
    if nun <= 80 and re.search(r"(^|[\s_\-])(year|yr)($|[\s_\-])", lname):
        if numeric:
            v = vals.astype(float)
            if v.min() >= 1800 and v.max() <= 2200 and (v % 1 == 0).all():
                return "year"
        elif sv.str.match(r"^\d{4}$").mean() >= 0.9:
            return "year"
    if numeric and re.search(r"(^|[\s_\-])month($|[\s_\-])", lname):
        v = vals.astype(float)
        if v.between(1, 12).all() and (v % 1 == 0).all():
            return "month_num"
    return None


def to_ordered_cat(s: pd.Series, kind: str) -> pd.Series:
    """Period column -> ordered categorical of strings (calendar / numeric order)."""
    notna = s.notna()
    if kind == "month_num":
        lab = s.where(notna, 0).astype(float).astype(int).map(lambda i: calendar.month_abbr[i] if 1 <= i <= 12 else None)
        cats = [calendar.month_abbr[i] for i in range(1, 13) if i in set(s.dropna().astype(int))]
    else:
        if pd.api.types.is_numeric_dtype(s):
            lab = s.where(notna, 0).astype(float).astype(int).astype(str)
            if kind == "quarter":
                lab = "Q" + lab
        else:
            lab = s.astype(str).str.strip()
        lab = lab.where(notna)
        uniq = list(dict.fromkeys(lab.dropna()))
        cats = sorted(uniq, key=(lambda v: _MONTHS.get(str(v).lower(), 99)) if kind == "month" else _nat)
    return pd.Series(pd.Categorical(lab, categories=cats, ordered=True), index=s.index)


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Text columns that really are dates -> datetime (by value pattern or column name)."""
    df = df.copy()
    pats = (r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", r"^\d{4}[-/]\d{1,2}$",
            r"\d{1,2}[\s\-/]+[A-Za-z]{3,9}[\s\-/,]+\d{2,4}", r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}")
    for c in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[c].dropna().astype(str).head(200)
        if sample.empty:
            continue
        pattern_hit = max(sample.str.contains(p, regex=True).mean() for p in pats) >= 0.9
        name_hit = bool(_DATE_HINT.search(str(c).lower())) and sample.str.contains(r"\d").mean() >= 0.9
        if not (pattern_hit or name_hit):
            continue
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", format="mixed")
        except Exception:
            parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.9:
            df[c] = parsed
    return df


def is_id_like(s: pd.Series, name: str) -> bool:
    if _cleaner_is_id is not None and _cleaner_is_id(name):
        return True
    if re.search(r"(^|[\s_\-])(id|uuid|guid|index|key)$", name.lower().strip()):
        return True
    return bool(pd.api.types.is_integer_dtype(s) and s.nunique() / max(len(s), 1) > 0.98
                and s.is_monotonic_increasing)


def prepare_frame(raw: pd.DataFrame):
    """-> (df, metrics, dims, dates, periods).  Generic cleaning for charting only."""
    df = raw.copy()
    df.columns = [str(c) for c in df.columns]
    num = df.select_dtypes("number").columns
    if len(num):
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    periods = {}
    for c in list(df.columns):
        kind = detect_period(df[c], c)
        if kind:
            df[c] = to_ordered_cat(df[c], kind)
            periods[c] = kind
    df = convert_dates(df)

    # small integer codes (Pclass 1/2/3, #children 0-5...) are groups, not measures.
    for c in list(df.columns):
        s = df[c]
        if c in periods or not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        vals = s.dropna()
        if vals.empty or is_id_like(s, c) or not (vals % 1 == 0).all():
            continue
        nun = vals.nunique()
        if nun == 2 and set(vals.unique()) <= {0, 1}:          # yes/no flag: keep as metric AND as a group
            df[f"{c} (flag)"] = s.map({1: "Yes", 0: "No"})
        elif 3 <= nun <= 8 and not _METRICISH.search(c.lower()):
            df[c] = pd.Categorical(s.astype("Int64").astype(str).where(s.notna()),
                                   categories=[str(v) for v in sorted(vals.astype(int).unique())], ordered=True)

    metrics, dims, dates = classify_columns(df, periods)
    for d in dims:                                              # missing groups -> "Unknown" so totals reconcile
        if d in periods:
            continue
        s = df[d]
        if s.isna().any():
            if isinstance(s.dtype, pd.CategoricalDtype):
                df[d] = s.cat.add_categories(UNKNOWN).fillna(UNKNOWN)
            else:
                df[d] = s.astype("object").where(s.notna(), UNKNOWN)
    return df, metrics, dims, dates, periods


def classify_columns(df: pd.DataFrame, periods=None):
    """-> (metrics, dims, dates). Dims may be high-cardinality (e.g. Course Name)."""
    periods = periods or {}
    metrics, dims, dates = [], [], []
    limit = min(1000, max(2, int(0.6 * len(df))))
    for c in df.columns:
        s = df[c]
        if s.isna().mean() > 0.95:
            continue
        if c in periods:
            if s.nunique() >= 2:
                dims.append(c)
        elif pd.api.types.is_datetime64_any_dtype(s):
            dates.append(c)
        elif pd.api.types.is_bool_dtype(s):
            if s.nunique() >= 2:
                dims.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            if s.nunique() > 1 and not is_id_like(s, c):
                metrics.append(c)
        elif 2 <= s.nunique() <= limit and not is_id_like(s, c):
            dims.append(c)
    return metrics, dims, dates


_MEAN_TOKENS = {"rate", "rates", "rating", "ratings", "score", "scores", "percent", "percentage", "pct", "ratio",
                "avg", "average", "mean", "age", "price", "prices", "margin", "satisfaction", "temperature", "temp",
                "duration", "fare", "salary", "wage", "income", "weight", "height", "speed", "bmi", "gpa"}
_SUM_TOKENS = {"amount", "revenue", "sales", "sale", "profit", "cost", "costs", "spend", "spending", "units", "unit",
               "quantity", "qty", "count", "total", "enrollments", "enrollment", "enrolled", "completions", "completed",
               "certificates", "issued", "views", "clicks", "orders", "volume", "sum", "visits", "downloads", "sold"}


def default_agg(name: str, s=None) -> str:
    """Sum for volumes/amounts, mean for ratings/rates/measurements. Uses the data when the name is unclear."""
    tokens = set(re.findall(r"[a-z]+", str(name).lower()))
    if tokens & _MEAN_TOKENS:
        return "mean"
    if tokens & _SUM_TOKENS:
        return "sum"
    if s is not None:
        v = pd.Series(s).dropna()
        if len(v) and (not (v % 1 == 0).all() or (v < 0).any()):   # decimals / signed values = a measurement
            return "mean"
    return "sum"


def order_metrics(metrics):
    return sorted(metrics, key=lambda c: (0 if any(k in c.lower() for k in _PREF) else 1, metrics.index(c)))


def best_period(periods, df):
    cands = [c for c in periods if df[c].nunique() >= 3] or [c for c in periods if df[c].nunique() >= 2]
    return min(cands, key=lambda c: (_PERIOD_ORDER.index(periods[c]), -df[c].nunique())) if cands else None


def eta(cat: pd.Series, y: pd.Series) -> float:
    g = pd.DataFrame({"c": cat.astype(str), "y": y}).dropna()
    ss_t = ((g["y"] - g["y"].mean()) ** 2).sum()
    if g.empty or ss_t == 0:
        return 0.0
    ss_b = g.groupby("c")["y"].apply(lambda v: len(v) * (v.mean() - g["y"].mean()) ** 2).sum()
    return float(np.sqrt(ss_b / ss_t))


def label_for(metric, agg):
    word = {"mean": "Average", "median": "Median", "count": "Count of", "nunique": "Unique"}.get(agg, "Total")
    return f"{word} {metric}"


# ---------------------------------------------------------- profile (safe to send to an LLM)
def build_profile(df, metrics, dims, dates, periods=None):
    """Schema + summary statistics only. No raw rows are ever included."""
    periods = periods or {}
    return {
        "rows": int(len(df)),
        "metrics": {m: {"min": float(df[m].min()), "max": float(df[m].max()),
                        "mean": round(float(df[m].mean()), 4)} for m in metrics},
        "dimensions": {d: {"unique": int(df[d].nunique()),
                           "top_values": [str(v) for v in df[d].value_counts().head(5).index],
                           **({"period_type": periods[d]} if d in periods else {})} for d in dims},
        "date_columns": {c: {"from": str(df[c].min())[:10], "to": str(df[c].max())[:10]} for c in dates},
    }


# ---------------------------------------------------------- rule-based planner
def plan_rules(df, metrics, dims, dates, periods=None, focus=None):
    periods = periods or {}
    ordered = order_metrics(metrics)
    if focus in ordered:
        ordered = [focus] + [m for m in ordered if m != focus]
    m1 = ordered[0]
    m2 = ordered[1] if len(ordered) > 1 else None
    agg1 = default_agg(m1, df[m1])
    verb = "Total" if agg1 == "sum" else "Average"

    plain = [d for d in dims if d not in periods]
    nun = {d: int(df[d].nunique()) for d in dims}
    strength = {d: eta(df[d], df[m1]) for d in plain if nun[d] <= 60}
    ranked = sorted(strength, key=lambda d: -strength[d])
    low = [d for d in ranked if 2 <= nun[d] <= 12]
    entity = sorted([d for d in plain if nun[d] >= 20], key=lambda d: -nun[d])
    mid = [d for d in ranked if nun[d] >= 5]
    top_dim = entity[0] if entity else (mid[0] if mid else (ranked[0] if ranked else None))
    dim2 = next((d for d in (mid + ranked) if d != top_dim), None)
    pcol = None if dates else best_period(periods, df)

    kpis = []
    if entity:
        kpis.append({"label": f"Unique {entity[0]}", "metric": entity[0], "agg": "nunique"})
    for m in ordered:
        if len(kpis) == 4:
            break
        a = default_agg(m, df[m])
        kpis.append({"label": label_for(m, a), "metric": m, "agg": a})
    if len(kpis) < 3:
        kpis.append({"label": f"Median {m1}", "metric": m1, "agg": "median"})
    if len(kpis) < 3:
        kpis.append({"label": "Records", "metric": None, "agg": "count"})

    charts = []

    def add(**k):
        k.setdefault("metric2", None); k.setdefault("dimension", None); k.setdefault("dimension2", None)
        k.setdefault("agg", agg1); k.setdefault("n", 10); k.setdefault("why", ""); k.setdefault("period", None)
        charts.append(k)

    if top_dim:
        add(type="top_bar", dimension=top_dim, metric=m1, title=f"Top {top_dim} by {verb.lower()} {m1}",
            why="Shows which items matter most")
    if dates or pcol:
        seasonal = bool(pcol) and periods[pcol] in ("month", "quarter", "month_num")   # no year => a pattern, not a timeline
        add(type="trend", metric=m1, metric2=m2, period=pcol,
            title=(f"{verb} {m1} by {pcol}" if seasonal else f"{verb} {m1} over time"),
            why="Shows the pattern across periods" if seasonal else "Shows momentum and spikes")
    elif dim2:
        add(type="column_rank", dimension=dim2, metric=m1, agg="mean",
            title=f"Average {m1} by {dim2}", why="Compares groups fairly")
    add(type="drivers", metric=m1, title=f"What influences {m1}", why="Ranks factors by influence")
    if low:
        add(type="donut", dimension=low[0], metric=m1, agg="sum" if agg1 == "sum" else "count",
            title=(f"Share of {verb.lower()} {m1} by {low[0]}" if agg1 == "sum" else f"Record share by {low[0]}"),
            why="Shows how the total splits")
    stack_dim = low[1] if len(low) > 1 else (low[0] if low else None)
    if dates and stack_dim:
        add(type="stacked", dimension=stack_dim, metric=m1, title=f"{verb} {m1} by period and {stack_dim}",
            why="Shows how the mix changes over time")
    elif pcol and stack_dim:
        add(type="stacked", dimension=stack_dim, dimension2=pcol, metric=m1,
            title=f"{verb} {m1} by {pcol} and {stack_dim}", why="Shows how the mix changes over time")
    elif len(low) > 1:
        add(type="stacked", dimension=low[1], dimension2=low[0], metric=m1,
            title=f"{verb} {m1}: {low[0]} by {low[1]}", why="Shows how two groupings combine")
    if dates:
        add(type="seasonality", metric=m1, title=f"{verb} {m1} by month of year", why="Reveals seasonal peaks")
    if dim2 and (dates or pcol):
        add(type="column_rank", dimension=dim2, metric=m1, agg="mean",
            title=f"Average {m1} by {dim2}", why="Compares groups fairly")
    if len(low) > 1:
        add(type="heatmap", dimension=low[0], dimension2=low[1], metric=m1, agg="mean",
            title=f"Average {m1}: {low[0]} x {low[1]}", why="Finds the best and worst combinations")
    if m2:
        add(type="scatter", metric=m1, metric2=m2, dimension=low[0] if low else None,
            title=f"{m2} vs {m1}", why="Tests whether two measures move together")
    add(type="distribution", metric=m1, title=f"Distribution of {m1}", why="Shows spread and outliers")
    if low:
        add(type="box", dimension=low[0], metric=m1, title=f"{m1} spread by {low[0]}",
            why="Shows variation inside each group")

    slicers = ([pcol] if pcol and 2 <= nun.get(pcol, 0) <= 40 else []) + [d for d in ranked if 2 <= nun[d] <= 40]
    return {"source": "rules", "headline": "", "primary": m1, "kpis": kpis[:4],
            "slicers": slicers[:3], "charts": charts}


# ---------------------------------------------------------- Gemini planner
def build_prompt(profile: dict) -> str:
    menu = "\n".join(f'  - "{k}": {v}' for k, v in CHART_TYPES.items())
    return f"""You are a senior BI analyst designing a Power BI-style executive dashboard.

DATA PROFILE (schema and summary statistics only):
{json.dumps(profile, indent=1, default=str)}

Return ONLY valid JSON in exactly this shape:
{{"headline": "<one sentence: what this dataset is about>",
  "primary_metric": "<one metric column>",
  "kpis": [{{"label": "<short>", "metric": "<column>", "agg": "sum|mean|median|count|nunique"}}],
  "slicers": ["<dimension column>"],
  "charts": [{{"type": "<chart type>", "title": "<short>", "metric": "<metric column or null>",
               "metric2": null, "dimension": null, "dimension2": null,
               "agg": "sum|mean|median|count", "n": 10, "why": "<max 15 words>"}}]}}

CHART TYPES (use only these):
{menu}

RULES
- Use ONLY column names that appear in the profile. Metrics go in metric/metric2, dimensions in dimension/dimension2.
- kpis: exactly 4. Use mean for ratings/rates/scores/ages, sum for volumes and amounts, nunique on a dimension for "how many distinct X".
- slicers: at most 3 dimensions with 2-40 unique values.
- charts: 6 to 9, no two alike. Most important first. Mix rankings, trend/seasonality (only if a date or period column exists), share, relationships and one drivers chart.
- Pick charts that reveal business insight. Never use identifier-like columns."""


def parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.I | re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def call_gemini(api_key: str, model: str, prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=api_key)
    r = client.models.generate_content(
        model=model, contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.2})
    return r.text


def validate_plan(raw: dict, df, metrics, dims, dates, periods, fallback: dict):
    """Keep only what refers to real columns; top up from the rule plan if needed."""
    periods = periods or {}
    notes, charts = [], []
    nun = {d: int(df[d].nunique()) for d in dims}
    pcol = None if dates else best_period(periods, df)

    for c in (raw.get("charts") or []):
        if not isinstance(c, dict) or c.get("type") not in CHART_TYPES:
            notes.append(f"dropped unknown chart {c!r:.60}"); continue
        t = c["type"]
        m = c.get("metric") if c.get("metric") in metrics else None
        agg = c.get("agg") if c.get("agg") in AGGS else default_agg(m or "", df[m] if m else None)
        n = c.get("n")
        spec = {"type": t, "title": str(c.get("title") or "")[:80], "metric": m,
                "metric2": c.get("metric2") if c.get("metric2") in metrics else None,
                "dimension": c.get("dimension") if c.get("dimension") in dims else None,
                "dimension2": c.get("dimension2") if c.get("dimension2") in dims else None,
                "agg": agg, "n": int(min(15, max(5, n))) if isinstance(n, (int, float)) else 10,
                "why": str(c.get("why") or "")[:120], "period": None}
        if m is None and t in ("top_bar", "column_rank", "trend", "donut", "stacked", "seasonality"):
            spec["agg"] = "count"
        d1, d2 = spec["dimension"], spec["dimension2"]
        if t == "trend" and not dates and pcol:
            spec["period"] = pcol
        if t == "stacked" and not d2 and not dates and pcol:
            spec["dimension2"] = d2 = pcol
        ok = {
            "top_bar": d1 and nun.get(d1, 0) >= 3,
            "column_rank": d1 and nun.get(d1, 0) >= 3,
            "trend": bool(dates or pcol),
            "donut": d1 and 2 <= nun.get(d1, 0) <= 40,
            "stacked": d1 and (d2 or dates),
            "seasonality": bool(dates),
            "heatmap": d1 and d2 and d1 != d2 and m,
            "scatter": m and spec["metric2"] and m != spec["metric2"],
            "distribution": m,
            "box": d1 and m,
            "drivers": True,
        }[t]
        if not ok:
            notes.append(f"dropped invalid {t}"); continue
        if t == "donut" and spec["agg"] in ("mean", "median"):
            spec["agg"] = "count"
        if any(spec["type"] == k["type"] and spec["dimension"] == k["dimension"] and spec["metric"] == k["metric"]
               and spec["dimension2"] == k["dimension2"] for k in charts):
            continue
        charts.append(spec)

    kpis = []
    for k in (raw.get("kpis") or []):
        if not isinstance(k, dict):
            continue
        a, m = k.get("agg"), k.get("metric")
        if a == "nunique" and m in dims:
            kpis.append({"label": str(k.get("label") or f"Unique {m}")[:40], "metric": m, "agg": a})
        elif a in AGGS and m in metrics:
            kpis.append({"label": str(k.get("label") or label_for(m, a))[:40], "metric": m, "agg": a})
        elif a == "count" and m is None:
            kpis.append({"label": str(k.get("label") or "Records")[:40], "metric": None, "agg": "count"})
    if len(kpis) < 3:
        notes.append("kpis topped up from rules"); kpis = (kpis + fallback["kpis"])[:4]

    if len(charts) < 4:
        notes.append("charts topped up from rules")
        have = {(c["type"], c["dimension"]) for c in charts}
        charts += [c for c in fallback["charts"] if (c["type"], c["dimension"]) not in have]

    slicers = [s for s in (raw.get("slicers") or []) if s in dims and 2 <= nun[s] <= 40][:3] or fallback["slicers"]
    primary = raw.get("primary_metric") if raw.get("primary_metric") in metrics else fallback["primary"]
    return {"source": "gemini", "headline": str(raw.get("headline") or "")[:200], "primary": primary,
            "kpis": kpis[:4], "slicers": slicers, "charts": charts[:12], "notes": notes}