
import pandas as pd
import numpy as np


# ==========================================================
# 1. DATASET CONTEXT BUILDER
#    Produces a plain-text summary injected into every
#    Gemini prompt as grounding context.
# ==========================================================

def build_dataset_context(df: pd.DataFrame, max_sample_rows: int = 5) -> str:
    """
    Return a comprehensive plain-text summary of *df* suitable
    for use as the system/context block in a Gemini prompt.
    """
    lines = []

    # --- shape ---
    lines.append(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- classify columns ---
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    text_cols     = df.select_dtypes(
                        include=["object", "category", "string"]
                    ).columns.tolist()
    dt_cols       = df.select_dtypes(include="datetime").columns.tolist()
    other_cols    = [c for c in df.columns
                     if c not in numeric_cols + text_cols + dt_cols]

    lines.append(
        f"Columns: {len(numeric_cols)} numeric · "
        f"{len(text_cols)} text/categorical · "
        f"{len(dt_cols)} datetime · "
        f"{len(other_cols)} other"
    )

    # --- per-column details ---
    lines.append("\n── Column Details ──")
    for col in df.columns:
        series       = df[col]
        dtype        = str(series.dtype)
        n_missing    = int(series.isna().sum())
        n_unique     = int(series.nunique())
        pct_missing  = n_missing / len(df) * 100 if len(df) else 0

        detail = f"  {col} [{dtype}]  {n_unique} unique"
        if n_missing:
            detail += f"  {n_missing} missing ({pct_missing:.1f}%)"

        if col in numeric_cols:
            vals = series.dropna()
            if len(vals):
                detail += (
                    f"  min={vals.min():.4g}"
                    f"  max={vals.max():.4g}"
                    f"  mean={vals.mean():.4g}"
                    f"  median={vals.median():.4g}"
                    f"  std={vals.std():.4g}"
                )
        elif col in text_cols:
            top = series.value_counts().head(5)
            top_str = "  |  ".join(f"'{v}' ({c})" for v, c in top.items())
            detail += f"  top → {top_str}"

        lines.append(detail)

    # --- numeric describe ---
    if numeric_cols:
        lines.append("\n── Numeric Summary (describe) ──")
        lines.append(df[numeric_cols].describe().round(4).to_string())

    # --- strong correlations ---
    if len(numeric_cols) >= 2:
        corr  = df[numeric_cols].corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        strong = (
            upper.stack()
            .reset_index()
            .rename(columns={"level_0": "A", "level_1": "B", 0: "r"})
            .query("r >= 0.70")
            .sort_values("r", ascending=False)
            .head(5)
        )
        if not strong.empty:
            lines.append("\n── Strong Correlations (|r| ≥ 0.70) ──")
            for _, row in strong.iterrows():
                lines.append(f"  {row['A']} ↔ {row['B']}  r={row['r']:.3f}")

    # --- data sample ---
    lines.append(f"\n── First {max_sample_rows} Rows ──")
    lines.append(df.head(max_sample_rows).to_string(index=False))

    return "\n".join(lines)


# ==========================================================
# 2. STAT HELPERS  (used by AI Analyst cards)
# ==========================================================

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """describe() extended with skewness and kurtosis."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["skewness"] = num.skew().values
    desc["kurtosis"] = num.kurt().values
    return desc.round(4)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns with missing values, sorted descending."""
    miss = df.isna().sum().reset_index()
    miss.columns = ["Column", "Missing Count"]
    miss["Missing %"] = (miss["Missing Count"] / len(df) * 100).round(2)
    return (
        miss[miss["Missing Count"] > 0]
        .sort_values("Missing Count", ascending=False)
        .reset_index(drop=True)
    )


def top_correlations(
    df: pd.DataFrame,
    threshold: float = 0.5,
    top_n: int = 10
) -> pd.DataFrame:
    """Return top correlated column pairs above *threshold*."""
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return pd.DataFrame()
    corr  = num.corr(numeric_only=True)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "Column A", "level_1": "Column B", 0: "Correlation"})
        .assign(Abs=lambda x: x["Correlation"].abs())
        .query(f"Abs >= {threshold}")
        .sort_values("Abs", ascending=False)
        .drop(columns="Abs")
        .head(top_n)
        .reset_index(drop=True)
    )
    pairs["Correlation"] = pairs["Correlation"].round(4)
    return pairs


def outlier_summary_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """IQR-based outlier count per numeric column."""
    rows = []
    for col in df.select_dtypes(include="number").columns:
        vals = df[col].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((vals < lower) | (vals > upper)).sum())
        rows.append({
            "Column": col,
            "Outliers": n,
            "Outlier %": round(n / len(vals) * 100, 2),
            "Lower Fence": round(lower, 4),
            "Upper Fence": round(upper, 4),
        })
    return pd.DataFrame(rows).sort_values("Outliers", ascending=False)
