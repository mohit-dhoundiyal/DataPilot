import calendar
import hashlib
import inspect
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from engine.dashboard_planner import (
    UNKNOWN, best_period, build_profile, build_prompt, call_gemini, default_agg, eta,
    order_metrics, parse_json, plan_rules, prepare_frame, validate_plan,
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="DataPilot | Dashboard", page_icon="📊", layout="wide")

from utils.ui_styles import inject_css
inject_css()

_W = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters \
    else {"use_container_width": True}

INDIGO, PALETTE = "#4F46E5", ["#4F46E5", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EC4899", "#64748B", "#EF4444"]
GREY, AMBER, RED = "#CBD5E1", "#F59E0B", "#EF4444"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
.dp-banner {background: linear-gradient(135deg,#312E81 0%,#4F46E5 70%,#6366F1 100%); padding: 18px 26px;
    border-radius: 14px; color:#fff; margin-bottom: 14px; text-align:center;}
.dp-banner h1 {margin:0; font-size:1.7rem; color:#fff; padding:0; font-family: Georgia, serif;}
.dp-banner p {margin:4px 0 0 0; opacity:.9; font-size:.92rem;}
.dp-kpi {background: rgba(79,70,229,0.07); border: 1px solid rgba(79,70,229,0.28); border-radius: 14px;
    padding: 12px 16px; display:flex; align-items:center; gap:12px; height:100%;}
.dp-kpi .ic {font-size:1.7rem; background: rgba(79,70,229,0.15); border-radius:50%; width:46px; height:46px;
    display:flex; align-items:center; justify-content:center;}
.dp-kpi .lbl {font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; opacity:.75; font-weight:600;}
.dp-kpi .val {font-size:1.55rem; font-weight:800; line-height:1.2; color:#4F46E5;}
.dp-kpi .sub {font-size:.78rem; opacity:.7;}
.up {color:#10B981; font-weight:600;} .down {color:#EF4444; font-weight:600;}
.dp-insight {background: rgba(79,70,229,0.07); border-left: 4px solid #4F46E5; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 8px;}
.dp-insight .t {font-weight:700; font-size:.9rem;} .dp-insight .b {font-size:.88rem; opacity:.9; line-height:1.4;}
.dp-h {font-size:1.05rem; font-weight:700; margin: 10px 0 4px 0;}
</style>""", unsafe_allow_html=True)


# ==========================================================
# HELPERS
# ==========================================================

def fmt(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if np.isnan(v):
        return "–"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:.2f}{suf}"
    return f"{int(v):,}" if v == int(v) else f"{v:,.2f}"


def icon_for(label):
    l = label.lower()
    for keys, ic in ((("rating", "score", "satisf"), "⭐"), (("rate", "percent", "%"), "📈"),
                     (("unique", "distinct"), "🗂️"), (("amount", "revenue", "sales", "price", "cost", "profit"), "💰"),
                     (("enroll", "customer", "user", "member"), "👥")):
        if any(k in l for k in keys):
            return ic
    return "📊"


def kpi_card(label, value, sub="", delta=None):
    d = ""
    if delta is not None and not np.isnan(delta):
        d = f'<span class="{"up" if delta >= 0 else "down"}">{"▲" if delta >= 0 else "▼"} {abs(delta):.1%}</span> '
    return (f'<div class="dp-kpi"><div class="ic">{icon_for(label)}</div><div><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{d}{sub}</div></div></div>')


def insight(icon, title, body):
    return f'<div class="dp-insight"><div class="t">{icon} {title}</div><div class="b">{body}</div></div>'


def heading(t):
    st.markdown(f'<div class="dp-h">{t}</div>', unsafe_allow_html=True)


def short_labels(vals, n=24):
    out, seen = [], {}
    for v in map(str, vals):
        s = v if len(v) <= n else v[:n - 1] + "…"
        seen[s] = seen.get(s, 0) + 1
        out.append(s if seen[s] == 1 else f"{s} ({seen[s]})")
    return out


def wrap_title(t, n=36):
    lines = [""]
    for w in t.split():
        if lines[-1] and len(lines[-1]) + len(w) + 1 > n:
            lines.append(w)
        else:
            lines[-1] = (lines[-1] + " " + w).strip()
    return "<br>".join(lines[:3])


def style(fig, title, height=330):
    title = wrap_title(title)
    fig.update_layout(template="plotly_white", height=height,
                      margin=dict(l=8, r=8, t=46 + 16 * title.count("<br>"), b=8),
                      title=dict(text=title, font=dict(size=13, color="#4338CA")),
                      font=dict(size=11), showlegend=fig.layout.showlegend if fig.layout.showlegend is not None else False)
    return fig


def agg_by(d, by, metric, agg):
    g = d.groupby(by, observed=True)
    return g.size() if (agg == "count" or metric is None) else g[metric].agg(agg)


def pick_freq(d, tcol):
    span = (d[tcol].max() - d[tcol].min()).days
    return (("D", "Daily") if span <= 60 else ("W", "Weekly") if span <= 365
            else ("MS", "Monthly") if span <= 1460 else ("QS", "Quarterly"))


def period_series(d, tcol, metric, agg, freq):
    r = d.dropna(subset=[tcol]).set_index(tcol).sort_index()
    s = r.resample(freq).size() if (agg == "count" or metric is None) else r[metric].resample(freq).agg(agg)
    return s.dropna()


def period_groups(d, pcol, metric, agg):
    """Aggregate by a period column (month names, FY, year...) in natural order."""
    g = d.groupby(pcol, observed=True)
    s = g.size() if (agg == "count" or metric is None) else g[metric].agg(agg)
    return s.dropna()


def spec_name(spec):
    return spec["metric"] or "records"


def kpi_value(d, k):
    m, a = k["metric"], k["agg"]
    if a == "nunique":
        col = d[m]
        return col[col.astype(str) != UNKNOWN].nunique()
    if a == "count":
        return len(d) if m is None else d[m].count()
    s = d[m].dropna()
    return getattr(s, a)() if len(s) else np.nan


# ==========================================================
# DATA (cached)
# ==========================================================

@st.cache_data(show_spinner=False)
def prep(raw: pd.DataFrame):
    return prepare_frame(raw)


@st.cache_data(show_spinner=False)
def cached_rules(_df, data_sig, focus, metrics, dims, dates, periods):
    return plan_rules(_df, metrics, dims, dates, periods, focus=focus)


@st.cache_data(show_spinner="Finding key drivers…")
def compute_importance(data: pd.DataFrame, target: str, features: tuple):
    features = list(features)
    X = data[features + [target]].dropna(subset=[target])
    if len(X) < 50 or not features:
        return pd.DataFrame(), None
    if len(X) > 10000:
        X = X.sample(10000, random_state=42)
    y = X[target].astype(float).values
    enc, kind = pd.DataFrame(index=X.index), {}
    for f in features:
        s = X[f]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            enc[f], kind[f] = s.fillna(s.median()), "Numeric"
        else:
            enc[f], kind[f] = pd.factorize(s.astype(str))[0], "Categorical"
    imp, r2 = None, None
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import GroupShuffleSplit
        key = pd.DataFrame({f: (X[f].round(0) if kind[f] == "Numeric" else X[f].astype(str)) for f in features})
        groups = pd.util.hash_pandas_object(key.astype(str), index=False).values
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(enc, y, groups))
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, n_jobs=-1,
                                      random_state=42).fit(enc.iloc[tr], y[tr])
        r2 = float(model.score(enc.iloc[te], y[te]))
        pi = permutation_importance(model, enc.iloc[te].iloc[:3000], y[te][:3000], n_repeats=3,
                                    random_state=42, n_jobs=-1)
        imp = np.clip(pi.importances_mean, 0, None)
        if imp.sum() == 0:
            imp = None
    except Exception:
        pass
    if imp is None:
        ys = pd.Series(y, index=X.index)
        imp = np.nan_to_num([abs(np.corrcoef(enc[f], y)[0, 1]) if kind[f] == "Numeric" and enc[f].nunique() > 1
                             else eta(X[f], ys) if kind[f] == "Categorical" else 0 for f in features])
    tot = imp.sum()
    out = pd.DataFrame({"Factor": features, "Importance": imp / tot * 100 if tot else imp})
    return out.sort_values("Importance", ascending=False).reset_index(drop=True), r2


# ==========================================================
# LOAD
# ==========================================================

source = st.session_state.get("cleaned_df")
if source is None:
    source = st.session_state.get("df")
    if source is None:
        st.warning("Please upload a CSV or Excel dataset from the Home page first.")
        st.stop()
    _dups, _miss = int(source.duplicated().sum()), int(source.isna().sum().sum())
    _issues = [x for x in (f"{_dups:,} duplicate rows" if _dups else "", f"{_miss:,} missing values" if _miss else "") if x]
    st.info("Showing the original dataset" + (f" ({' and '.join(_issues)} not cleaned yet)" if _issues else "")
            + ". Run **Data Cleaning** first for more reliable results.")
if source.empty:
    st.warning("The dataset is empty.")
    st.stop()

df, metrics, dims, dates, periods = prep(source)
if not metrics:
    st.error("No usable numeric column was found, so a KPI dashboard cannot be built.")
    st.stop()
time_col = dates[0] if dates else None
pcol = None if dates else best_period(periods, df)
sig = hashlib.md5(f"{df.shape}{list(df.columns)}{list(map(str, df.dtypes))}".encode()).hexdigest()[:12]
data_sig = sig + str(round(float(np.nansum(df[metrics].to_numpy(dtype=float))), 3))


# ==========================================================
# SIDEBAR: layout mode (rules or Gemini)
# ==========================================================

st.sidebar.header("🎛️ Dashboard")


def find_key():
    k = st.session_state.get("gemini_api_key")
    if k:
        return k
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if st.secrets.get(name):
                return st.secrets[name]
        except Exception:
            pass
        if os.getenv(name):
            return os.getenv(name)
    return None


def find_model():
    try:
        if st.secrets.get("GEMINI_MODEL"):
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


mode = st.sidebar.radio("Chart layout", ["Smart rules (instant)", "✨ Gemini designer"], key="dash_mode",
                        help="Rules work offline and cost nothing. Gemini designs the layout once per dataset; "
                             "only column names and summary statistics are sent, never your rows.")
rules_focus = None
if mode.startswith("Smart"):
    rules_focus = st.sidebar.selectbox("Focus metric", order_metrics(metrics), key="dash_focus")

plan = None
gem_store = st.session_state.get("dash_gemini_plan") or {}
if mode.startswith("✨"):
    key = find_key()
    if not key:
        key = st.sidebar.text_input("Gemini API key", type="password", key="dash_key_input",
                                    help="Free key from Google AI Studio. Kept only in this session.")
        if key:
            st.session_state["gemini_api_key"] = key
    fallback = cached_rules(df, data_sig, None, metrics, dims, dates, periods)
    if key and st.sidebar.button("✨ Design layout with Gemini", use_container_width=True):
        try:
            with st.spinner("Gemini is designing your dashboard…"):
                raw = parse_json(call_gemini(key, find_model(), build_prompt(build_profile(df, metrics, dims, dates, periods))))
                st.session_state["dash_gemini_plan"] = {"sig": sig, "plan": validate_plan(raw, df, metrics, dims, dates, periods, fallback)}
                gem_store = st.session_state["dash_gemini_plan"]
        except Exception as e:  # quota, network, bad JSON...
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                st.sidebar.warning("Free-tier quota limit reached. Using the smart-rules layout instead.")
            elif "404" in msg or "not found" in msg.lower():
                st.sidebar.warning("Model not found. Please check GEMINI_MODEL in Secrets.")
            else:
                st.sidebar.warning(f"Gemini call failed ({msg[:60]}). Using smart-rules layout instead.")
    if gem_store.get("sig") == sig:
        plan = gem_store["plan"]
        st.sidebar.success("Layout designed by Gemini")
    elif key:
        st.sidebar.info("Click the button to design the layout. Smart rules are shown meanwhile.")
if plan is None:
    plan = cached_rules(df, data_sig, rules_focus, metrics, dims, dates, periods)
focus = plan["primary"]


# ==========================================================
# SLICERS (top of page, Power BI style)
# ==========================================================

st.markdown(f'<div class="dp-banner"><h1>Business Analytics Dashboard</h1>'
            f'<p>{plan.get("headline") or "Key performance, rankings, trends and what drives them"}</p></div>',
            unsafe_allow_html=True)

filtered = df
slicer_cols = st.columns(max(1, len(plan["slicers"]) + (1 if time_col else 0)))
for col, s in zip(slicer_cols, plan["slicers"]):
    with col:
        col_ = df[s]
        opts = ["All"] + ([str(v) for v in col_.cat.categories] if isinstance(col_.dtype, pd.CategoricalDtype) and col_.dtype.ordered
                          else sorted(col_.dropna().astype(str).unique()))
        pick = st.selectbox(s, opts, key=f"dash_s_{s}")
        if pick != "All":
            filtered = filtered[filtered[s].astype(str) == pick]
if time_col:
    with slicer_cols[len(plan["slicers"])]:
        years = ["All"] + sorted(df[time_col].dropna().dt.year.unique().astype(int).astype(str))
        yr = st.selectbox("Year", years, key="dash_year")
        if yr != "All":
            filtered = filtered[filtered[time_col].dt.year == int(yr)]

if len(filtered) < 10:
    st.warning("Too little data matches the current filters.")
    st.stop()

st.sidebar.download_button("⬇️ Download filtered data", filtered.to_csv(index=False).encode("utf-8"),
                           "datapilot_filtered.csv", "text/csv", use_container_width=True)


# ==========================================================
# KPI CARDS
# ==========================================================

growth, tlabel = None, ""
if time_col:
    freq, tlabel = pick_freq(filtered, time_col)
    ps = period_series(filtered, time_col, focus, default_agg(focus, df[focus]), freq)
elif pcol:
    ps = period_groups(filtered, pcol, focus, default_agg(focus, df[focus]))
if time_col or (pcol and periods[pcol] in ("fy", "year")):   # month/quarter alone is seasonality, not growth
    done = ps.iloc[:-1] if len(ps) >= 4 else ps          # latest period is usually incomplete
    if len(done) >= 2 and done.iloc[-2] and (ps >= 0).all():   # % change is meaningless for signed measures
        growth = (done.iloc[-1] - done.iloc[-2]) / abs(done.iloc[-2])
punit = f"{tlabel.lower()} period" if tlabel else "period"

kc = st.columns(len(plan["kpis"]))
growth_idx = next((i for i, k in enumerate(plan["kpis"]) if k["metric"] == focus and k["agg"] != "nunique"), None)
for i, (c, k) in enumerate(zip(kc, plan["kpis"])):
    is_first_metric = i == growth_idx
    c.markdown(kpi_card(k["label"], fmt(kpi_value(filtered, k)),
                        f"vs previous {punit}" if (growth is not None and is_first_metric) else "",
                        growth if is_first_metric else None), unsafe_allow_html=True)


# ==========================================================
# DRIVERS + TAKEAWAYS
# ==========================================================

feats = tuple([m for m in metrics if m != focus] + [d for d in dims if filtered[d].nunique() <= 60])
imp_df, r2 = compute_importance(filtered[list(feats) + [focus]], focus, feats) if feats else (pd.DataFrame(), None)
drivers_ok = not imp_df.empty and not (r2 is not None and r2 < 0.10)
modest = r2 is not None and 0.10 <= r2 < 0.30

top_spec = next((c for c in plan["charts"] if c["type"] == "top_bar"), None)


def takeaways():
    items, ser = [], filtered[focus].dropna()
    if growth is not None:
        items.append(("📈", "Trend", f"{focus} {'rose' if growth >= 0 else 'fell'} <b>{abs(growth):.1%}</b> in the latest "
                                    f"complete {punit}."))
    if top_spec and (ser >= 0).all():
        g = agg_by(filtered, top_spec["dimension"], focus, "sum").sort_values(ascending=False)
        if len(g) >= 4 and g.sum() > 0:
            s1, s3 = g.iloc[0] / g.sum(), g.iloc[:3].sum() / g.sum()
            if s1 >= max(0.15, 3 / len(g)) or s3 >= 0.5:
                items.append(("🧩", "Concentration", f"<b>{g.index[0]}</b> alone accounts for <b>{s1:.0%}</b> of {focus}; "
                                                      f"the top 3 {top_spec['dimension']} values make up <b>{s3:.0%}</b>."))
    if time_col and len(filtered) > 30:
        mo = filtered.dropna(subset=[time_col]).groupby(filtered[time_col].dt.month)[focus].agg(default_agg(focus, df[focus]))
        if len(mo) >= 6 and mo.mean():
            pk = mo.idxmax()
            lift = mo.max() / mo.mean() - 1
            if lift >= 0.25:
                items.append(("🗓️", "Seasonality", f"<b>{calendar.month_name[pk]}</b> is the peak month, "
                                                    f"<b>{lift:.0%}</b> above the average month."))
    if not imp_df.empty:
        if drivers_ok:
            names = ", ".join(f"<b>{r.Factor}</b> ({r.Importance:.0f}%)" for r in imp_df.head(3).itertuples())
            note = (f" They explain only ~{r2:.0%} of the variation, so treat this as a hint.") if modest else ""
            items.append(("🎯", "What moves it most", names + "." + note))
        else:
            items.append(("🎯", "Drivers", f"No factor explains {focus} well (fit {r2:.2f}); differences are mostly random."))
    best = None
    for d in dims:
        n = filtered[d].nunique()
        if 2 <= n <= 12:
            g = filtered.groupby(d, observed=True)[focus].agg(["mean", "std", "count"])
            g = g[g["count"] >= max(30, int(0.02 * len(filtered)))]
            if len(g) >= 2 and ser.mean():
                hi, lo = g["mean"].idxmax(), g["mean"].idxmin()
                se = np.sqrt(np.nan_to_num(g.loc[hi, "std"]) ** 2 / g.loc[hi, "count"] + np.nan_to_num(g.loc[lo, "std"]) ** 2 / g.loc[lo, "count"])
                gap = (g.loc[hi, "mean"] - g.loc[lo, "mean"]) / abs(ser.mean())
                if gap >= 0.10 and g.loc[hi, "mean"] - g.loc[lo, "mean"] >= 3 * se and (best is None or gap > best[0]):
                    best = (gap, d, hi, lo, g.loc[hi, "mean"], g.loc[lo, "mean"])
    if best:
        items.append(("⚖️", "Biggest gap", f"In <b>{best[1]}</b>, <b>{best[2]}</b> averages {fmt(best[4])} vs {fmt(best[5])} for <b>{best[3]}</b>."))
    q1, q3 = ser.quantile(.25), ser.quantile(.75)
    if q3 > q1 and ((ser < q1 - 1.5 * (q3 - q1)) | (ser > q3 + 1.5 * (q3 - q1))).mean() > 0.05:
        items.append(("⚠️", "Watch out", "Many extreme values may skew averages."))
    if not items:
        items.append(("✅", "Steady", f"Nothing stands out: {focus} is evenly spread across the data."))
    return items[:4]


heading("💡 Key takeaways")
tk = takeaways()
cols = st.columns(2)
for i, (ic, t, b) in enumerate(tk):
    cols[i % 2].markdown(insight(ic, t, b), unsafe_allow_html=True)


# ==========================================================
# CHART RENDERERS
# ==========================================================

def render(spec, d):
    t, m, m2, dim, dim2, agg, n = (spec["type"], spec["metric"], spec["metric2"], spec["dimension"],
                                   spec["dimension2"], spec["agg"], spec.get("n", 10))
    verb = {"sum": "Total", "mean": "Average", "median": "Median", "count": "Count"}[agg]
    title = spec["title"] or f"{verb} {m or 'records'}"

    if t in ("top_bar", "column_rank"):
        s = agg_by(d, dim, m, agg).sort_values(ascending=False).head(n)
        lab = short_labels(s.index)
        if t == "top_bar":
            fig = go.Figure(go.Bar(x=s.values[::-1], y=lab[::-1], orientation="h", marker_color=INDIGO,
                                   text=[fmt(v) for v in s.values[::-1]], textposition="outside",
                                   customdata=[str(i) for i in s.index[::-1]], hovertemplate="%{customdata}: %{x:,.2f}<extra></extra>", cliponaxis=False))
            fig.update_xaxes(visible=False, range=[0, s.max() * 1.3])
        else:
            fig = go.Figure(go.Bar(x=lab, y=s.values, marker_color=PALETTE[1], text=[fmt(v) for v in s.values],
                                   textposition="outside", customdata=[str(i) for i in s.index],
                                   hovertemplate="%{customdata}: %{y:,.2f}<extra></extra>", cliponaxis=False))
            fig.update_yaxes(visible=False, range=[0, s.max() * 1.25])
            fig.update_xaxes(tickangle=-35)
        return style(fig, title)

    if t == "trend":
        pc = spec.get("period")
        if pc:
            s1 = period_groups(d, pc, m, agg)
            s2 = period_groups(d, pc, m2, default_agg(m2, df[m2])) if m2 else None
            lab, x1 = str(pc), [str(i) for i in s1.index]
            x2 = [str(i) for i in s2.index] if s2 is not None else None
        else:
            if not time_col:
                return None
            freq, lab = pick_freq(d, time_col)
            s1 = period_series(d, time_col, m, agg, freq)
            s2 = period_series(d, time_col, m2, default_agg(m2, df[m2]), freq) if m2 else None
            x1, x2 = s1.index, (s2.index if s2 is not None else None)
        if len(s1) < 2:
            return None
        sec = s2 is not None and len(s2) > 1 and (s2.max() > 3 * s1.max() or s1.max() > 3 * s2.max())
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x1, y=s1.values, name=m or "records", mode="lines+markers",
                                 line=dict(color=INDIGO, width=2.5), marker=dict(size=5)), secondary_y=False)
        if s2 is not None and len(s2) > 1:
            fig.add_trace(go.Scatter(x=x2, y=s2.values, name=m2, mode="lines", line=dict(color=AMBER, width=2, dash="dot")),
                          secondary_y=bool(sec))
        fig.update_layout(showlegend=s2 is not None, legend=dict(orientation="h", y=-0.2), hovermode="x unified")
        if pc:
            fig.update_xaxes(tickangle=-35)
        return style(fig, title if pc else f"{lab} trend: {title}")

    if t == "donut":
        s = agg_by(d, dim, m, agg).sort_values(ascending=False)
        s = s[s > 0]
        if len(s) < 2:
            return None
        if len(s) > 6:
            s = pd.concat([s.head(6), pd.Series({"Other": s.iloc[6:].sum()})])
        fig = go.Figure(go.Pie(labels=[str(i) for i in s.index], values=s.values, hole=0.55, sort=False,
                               marker=dict(colors=PALETTE), textinfo="percent"))
        fig.update_layout(showlegend=True, legend=dict(font=dict(size=10)))
        return style(fig, title)

    if t == "stacked":
        top_stack = d[dim].astype(str).value_counts().head(5).index
        dd = d.assign(_s=np.where(d[dim].astype(str).isin(top_stack), d[dim].astype(str), "Other"))
        if dim2:
            if isinstance(dd[dim2].dtype, pd.CategoricalDtype) and dd[dim2].dtype.ordered:
                xs = [c for c in dd[dim2].cat.categories if c in set(dd[dim2].dropna())][-24:]   # natural order
            else:
                xs = agg_by(dd, dim2, m, agg).sort_values(ascending=False).head(10).index
            dd = dd[dd[dim2].isin(xs)]
            pv = (dd.groupby([dim2, "_s"], observed=True).size() if (agg == "count" or m is None)
                  else dd.groupby([dim2, "_s"], observed=True)[m].agg(agg)).unstack(fill_value=0)
            xlab = short_labels(pv.index)
        else:
            if not time_col:
                return None
            freq, _ = pick_freq(dd, time_col)
            gr = dd.groupby([pd.Grouper(key=time_col, freq=freq), "_s"])
            pv = (gr.size() if (agg == "count" or m is None) else gr[m].agg(agg)).unstack(fill_value=0)
            xlab = pv.index
        fig = go.Figure([go.Bar(x=xlab, y=pv[c].values, name=str(c), marker_color=PALETTE[i % len(PALETTE)])
                         for i, c in enumerate(pv.columns)])
        fig.update_layout(barmode="stack", showlegend=True, legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        fig.update_xaxes(tickangle=-35)
        fig = style(fig, title)
        fig.update_layout(margin=dict(l=8, r=8, t=fig.layout.margin.t, b=120), legend=dict(orientation="h", y=-0.55, font=dict(size=10)))
        return fig

    if t == "seasonality":
        if not time_col:
            return None
        dd = d.dropna(subset=[time_col])
        s = (dd.groupby(dd[time_col].dt.month).size() if (agg == "count" or m is None)
             else dd.groupby(dd[time_col].dt.month)[m].agg(agg))
        if len(s) < 3:
            return None
        fig = go.Figure(go.Scatter(x=[calendar.month_abbr[i] for i in s.index], y=s.values, mode="lines+markers+text",
                                   text=[fmt(v) for v in s.values], textposition="top center",
                                   line=dict(color=INDIGO, width=2.5), textfont=dict(size=9)))
        fig.update_yaxes(visible=False, range=[0, s.max() * 1.2])
        return style(fig, title)

    if t == "heatmap":
        a = d[dim].astype(str).value_counts().head(10).index
        b = d[dim2].astype(str).value_counts().head(10).index
        sub = d[d[dim].astype(str).isin(a) & d[dim2].astype(str).isin(b)]
        pv = sub.pivot_table(index=sub[dim].astype(str), columns=sub[dim2].astype(str), values=m, aggfunc=agg)
        if pv.empty:
            return None
        fig = px.imshow(pv, text_auto=".3s", aspect="auto", color_continuous_scale="Purples")
        fig.update_layout(coloraxis_showscale=False)
        return style(fig, title)

    if t == "scatter":
        cols_ = [m, m2] + ([dim] if dim and d[dim].nunique() <= 8 else [])
        sc = d[cols_].dropna()
        sc = sc.sample(min(len(sc), 3000), random_state=42)
        r = sc[m].corr(sc[m2])
        fig = px.scatter(sc, x=m, y=m2, color=dim if len(cols_) == 3 else None, opacity=0.55,
                         color_discrete_sequence=PALETTE)
        if sc[m].nunique() > 1:
            k, b = np.polyfit(sc[m], sc[m2], 1)
            xr = np.array([sc[m].min(), sc[m].max()])
            fig.add_trace(go.Scatter(x=xr, y=k * xr + b, mode="lines", line=dict(color=RED, dash="dash"), showlegend=False))
        fig.update_layout(showlegend=len(cols_) == 3, legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        return style(fig, f"{title} (r = {r:.2f})")

    if t == "distribution":
        fig = px.histogram(d if len(d) <= 150000 else d.sample(150000, random_state=1), x=m, nbins=25, marginal="box", color_discrete_sequence=[INDIGO])
        return style(fig, title)

    if t == "box":
        top = d[dim].astype(str).value_counts().head(8).index
        sub = d[d[dim].astype(str).isin(top)]
        sub = sub if len(sub) <= 100000 else sub.sample(100000, random_state=1)
        fig = px.box(sub, x=sub[dim].astype(str), y=m, points=False, color_discrete_sequence=[INDIGO])
        fig.update_xaxes(title=None, tickangle=-30)
        return style(fig, title)

    if t == "drivers":
        if imp_df.empty:
            return None
        p = imp_df.head(6).iloc[::-1]
        fig = go.Figure(go.Bar(x=p["Importance"], y=short_labels(p["Factor"]), orientation="h", marker_color=INDIGO,
                               text=[f"{v:.0f}%" for v in p["Importance"]], textposition="outside"))
        fig.update_xaxes(visible=False, range=[0, p["Importance"].max() * 1.25])
        return style(fig, title)
    return None


# ==========================================================
# GRID
# ==========================================================

visible = [c for c in plan["charts"] if not (c["type"] == "drivers" and not drivers_ok)]
main, more = visible[:6], visible[6:10]


def grid(specs, prefix):
    for r in range(0, len(specs), 3):
        cols = st.columns(3)
        for j, (col, spec) in enumerate(zip(cols, specs[r:r + 3])):
            with col:
                with st.container(border=True):
                    try:
                        fig = render(spec, filtered)
                    except Exception as e:  # never let one chart break the page
                        fig = None
                        st.caption(f"⚠️ Could not draw '{spec['title']}': {type(e).__name__}")
                    if fig is not None:
                        st.plotly_chart(fig, key=f"{prefix}_{r + j}", **_W)
                    elif not spec.get("_err"):
                        st.caption(f"Not enough data for “{spec['title']}”.")


heading("📊 Dashboard")
grid(main, "main")
if more:
    with st.expander(f"➕ More charts ({len(more)})"):
        grid(more, "more")

with st.expander("ℹ️ How these charts were chosen"):
    st.caption(("Designed by **Gemini** from column names and summary statistics only (no rows were sent). "
                "All numbers are computed locally. ") if plan["source"] == "gemini" else
               "Chosen by DataPilot's built-in rules from each column's type, cardinality and relationship to the focus metric.")
    for c in visible:
        st.write(f"• **{c['title']}** — {c.get('why') or c['type']}")
    for n_ in plan.get("notes", []):
        st.caption(f"validation: {n_}")