import difflib
import re

import numpy as np
import pandas as pd


# ==========================================================
# 0. SETTINGS
# ==========================================================

# Text that really means "no value". These are converted to missing
# (only when the user keeps the placeholder option switched on).
PLACEHOLDER_TOKENS = {
    "", "-", "--", "---", "?", "na", "n/a", "n.a.", "nan",
    "null", "none", "nil", "missing", "#n/a", "not available"
}

# A conversion (text -> number/date) that would turn more than this
# percentage of the values into missing is NOT applied unless the user
# explicitly allows it. This is the main protection against silent data loss.
MAX_CONVERSION_LOSS_PCT = 5.0


# ==========================================================
# 1. COLUMN IDENTIFICATION
# ==========================================================

_ID_EXCLUDE = {
    "paid", "valid", "invalid", "android", "void", "avoid", "solid",
    "liquid", "fluid", "rapid", "humid", "hybrid", "grid", "bid",
    "kid", "did", "amid", "acid", "candid", "vivid", "rigid", "lucid"
}

_ID_COMPACT_NAMES = {
    "id", "customerid", "userid", "productid", "employeeid",
    "transactionid", "orderid", "invoiceid", "phone", "phonenumber",
    "mobilenumber", "contactnumber", "zipcode", "postalcode",
    "accountnumber", "rollnumber", "registrationnumber"
}

# Words that mean "identifier" on their own.
_ID_WORDS = {
    "id", "uuid", "guid", "identifier", "phone", "zip", "zipcode",
    "pincode", "postalcode", "sku", "ssn"
}

# "account", "order" ... are only identifiers when combined with these.
_ID_QUALIFIED_PREFIX = {
    "account", "invoice", "reference", "contact", "roll", "registration",
    "transaction", "order", "customer", "user", "product", "employee",
    "student", "mobile"
}
_ID_QUALIFIED_SUFFIX = {"id", "no", "number", "num", "code", "key"}


def is_identifier_column(column_name):
    """
    Identify columns that probably contain identifiers, codes,
    contact numbers or reference numbers.

    'CustomerID', 'order_id' and 'Phone' are identifiers.
    'Paid', 'Account Balance' and 'Contact Count' are NOT.
    """

    name = str(column_name).strip()
    compact = re.sub(r"[^a-z0-9]", "", name.lower())

    # CustomerID -> "Customer ID" so that the word 'id' is found.
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).lower()
    tokens = {t for t in re.split(r"[^a-z0-9]+", spaced) if t}

    if compact in _ID_COMPACT_NAMES:
        return True

    if tokens & _ID_WORDS:
        return True

    if tokens & _ID_QUALIFIED_PREFIX and tokens & _ID_QUALIFIED_SUFFIX:
        return True

    return (
        compact.endswith("id")
        and len(compact) > 2
        and compact not in _ID_EXCLUDE
    )


def is_text_column(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


# ==========================================================
# 2. PLACEHOLDERS ("-", "N/A", "null" ...) = HIDDEN MISSING VALUES
# ==========================================================

def _placeholder_mask(series):
    """True where a text value is really a placeholder for 'missing'."""

    if not is_text_column(series):
        return pd.Series(False, index=series.index)

    text = series.astype("string").str.strip().str.lower()

    return (series.notna() & text.isin(PLACEHOLDER_TOKENS)).astype(bool)


def count_placeholders(series):
    return int(_placeholder_mask(series).sum())


def normalize_placeholders(series):
    """Return the series with placeholder text replaced by real missing values."""

    mask = _placeholder_mask(series)

    if not mask.any():
        return series

    return series.mask(mask)


# ==========================================================
# 3. SAFE CONVERSION: TEXT -> NUMBER
# ==========================================================

def _clean_number_text(text):
    """'Rs.3,045.46' -> '3045.46' ; '(50)' -> '-50' ; '$ 12' -> '12'."""

    s = text.astype("string").str.strip()

    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    s = s.str.replace(
        r"(?i)^([+-]?)\s*(?:rs\.?|inr|usd|eur|gbp)\s*",
        r"\1",
        regex=True
    )

    s = s.str.replace(
        r"(?i)\s*(?:rs\.?|inr|usd|eur|gbp)$",
        "",
        regex=True
    )

    s = s.str.replace(r"[₹$€£¥]", "", regex=True)

    s = s.str.replace(r"[,\s_]", "", regex=True)

    return s


def safe_to_numeric(series):
    """
    Convert text to numbers WITHOUT silently losing values.

    Returns (converted_series, report). The report says how many values
    could not be read and shows examples, so the caller can decide.
    """

    placeholder = _placeholder_mask(series)

    text = _clean_number_text(series).mask(placeholder)

    converted = pd.to_numeric(text, errors="coerce")

    # Plain float64 (not the nullable Float64) keeps charts and maths simple.
    converted = pd.Series(
        converted.to_numpy(dtype="float64", na_value=np.nan),
        index=series.index,
        name=series.name
    )

    attempted = int(text.notna().sum())
    failed_mask = text.notna() & converted.isna()
    failed = int(failed_mask.sum())

    report = {
        "placeholders": int(placeholder.sum()),
        "attempted": attempted,
        "failed": failed,
        "failed_examples": (
            series[failed_mask].astype(str).unique().tolist()[:5]
        ),
        "loss_pct": (failed / attempted * 100) if attempted else 0.0
    }

    return converted, report


# ==========================================================
# 4. SAFE CONVERSION: TEXT -> DATE
# ==========================================================

_NUMERIC_DATE_PATTERN = r"^(\d{1,4})([/\-.])(\d{1,2})\2(\d{1,4})$"

# Dates outside this range are treated as unreadable. This also keeps every
# value inside what pandas can store (about 1677-2262) and rejects junk such
# as year 0001 produced when a lone month name like "July" is parsed.
MIN_VALID_DATE = pd.Timestamp("1900-01-01")
MAX_VALID_DATE = pd.Timestamp("2100-12-31")


def _within_valid_range(parsed):
    return parsed.where((parsed >= MIN_VALID_DATE) & (parsed <= MAX_VALID_DATE))


def safe_to_datetime(series, dayfirst=True):
    """
    Parse dates that are written in several formats in the same column.

    A single pd.to_datetime() call guesses ONE format from the first values
    and turns every row that does not fit into missing. Instead, values are
    grouped by their layout (separator + year position/length). For every
    group both day-first and month-first are tested and the order that
    parses the most rows is used. Groups where both orders fit every row
    are reported as ambiguous.

    Returns (datetime_series, report).
    """

    empty_report = {
        "groups": {},
        "ambiguous": [],
        "attempted": 0,
        "failed": 0,
        "failed_examples": [],
        "loss_pct": 0.0
    }

    if pd.api.types.is_datetime64_any_dtype(series):
        return series, empty_report

    placeholder = _placeholder_mask(series)

    s = (
        series.astype("string")
        .str.strip()
        .mask(placeholder)
        .reset_index(drop=True)
    )

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    parts = s.str.extract(_NUMERIC_DATE_PATTERN)

    first_len = parts[0].str.len().fillna(0).astype(int)
    last_len = parts[3].str.len().fillna(0).astype(int)
    matched = parts[0].notna()

    kind = pd.Series(None, index=s.index, dtype="object")
    kind[matched & (first_len == 4)] = "year-first"
    kind[matched & (first_len != 4) & (last_len == 4)] = "year-last-4"
    kind[matched & (first_len != 4) & (last_len == 2)] = "year-last-2"

    report = dict(empty_report)
    report["groups"] = {}
    report["ambiguous"] = []

    grouping = pd.DataFrame({"kind": kind, "sep": parts[1]}).dropna()

    for (kind_name, sep), rows in grouping.groupby(["kind", "sep"]).groups.items():

        if kind_name == "year-first":
            formats = [f"%Y{sep}%m{sep}%d"]

        else:
            year = "%Y" if kind_name == "year-last-4" else "%y"
            day_first = f"%d{sep}%m{sep}{year}"
            month_first = f"%m{sep}%d{sep}{year}"
            formats = (
                [day_first, month_first]
                if dayfirst
                else [month_first, day_first]
            )

        subset = s.loc[rows]

        parsed = {
            fmt: pd.to_datetime(subset, format=fmt, errors="coerce")
            for fmt in formats
        }

        parsed = {fmt: _within_valid_range(p) for fmt, p in parsed.items()}

        failures = {fmt: int(p.isna().sum()) for fmt, p in parsed.items()}

        # min() keeps the first (preferred) format when there is a tie.
        best = min(formats, key=lambda f: failures[f])

        ties = [f for f in formats if failures[f] == failures[best]]

        label = f"{kind_name} '{sep}'"

        if len(ties) > 1:
            report["ambiguous"].append(label)

        out.loc[rows] = _within_valid_range(parsed[best])

        report["groups"][label] = {
            "format": best,
            "rows": int(len(rows)),
            "failed": failures[best],
            "ambiguous": len(ties) > 1
        }

    # Dates with month names or times, e.g. "Jan 05, 2025", "2025-01-05 10:30"
    leftover = (
        s.notna()
        & kind.isna()
        & s.str.contains(r"\d", regex=True).fillna(False).astype(bool)
        & s.str.contains(r"[A-Za-z:]", regex=True).fillna(False).astype(bool)
    )

    if leftover.any():

        subset = s[leftover]

        try:
            parsed = pd.to_datetime(
                subset, errors="coerce", dayfirst=dayfirst,
                format="mixed", utc=True
            )
        except (TypeError, ValueError):
            parsed = pd.to_datetime(
                subset, errors="coerce", dayfirst=dayfirst, utc=True
            )

        parsed = _within_valid_range(parsed.dt.tz_localize(None))

        out.loc[subset.index] = parsed

        report["groups"]["text/time formats"] = {
            "format": "automatic",
            "rows": int(leftover.sum()),
            "failed": int(parsed.isna().sum()),
            "ambiguous": False
        }

    attempted = int(s.notna().sum())
    failed_mask = s.notna() & out.isna()
    failed = int(failed_mask.sum())

    out.index = series.index

    failed_mask.index = series.index

    report["attempted"] = attempted
    report["failed"] = failed
    report["failed_examples"] = (
        series[failed_mask].astype(str).unique().tolist()[:5]
    )
    report["loss_pct"] = (failed / attempted * 100) if attempted else 0.0

    return out, report


# ==========================================================
# 5. TYPE DETECTION
# ==========================================================

def _sample_values(series, size=1000):
    """Non-missing, non-placeholder values (random sample for big columns)."""

    values = series[series.notna() & ~_placeholder_mask(series)]

    if len(values) > size:
        values = values.sample(size, random_state=0)

    return values


def detect_date_candidate(series, column_name):

    if pd.api.types.is_datetime64_any_dtype(series):
        return True, "Already a date/time column"

    if is_identifier_column(column_name):
        return False, "Identifier-like column"

    if pd.api.types.is_numeric_dtype(series):
        return False, "Numeric column"

    sample = _sample_values(series)

    if sample.empty:
        return False, "No values available"

    name = str(column_name).lower()

    date_words = [
        "date", "time", "dob", "birth", "created",
        "updated", "signup", "joining", "purchase"
    ]

    name_suggests_date = any(word in name for word in date_words)

    # Text that is really a number (e.g. prices) must never become a date.
    _, number_report = safe_to_numeric(sample)

    if (
        number_report["attempted"]
        and number_report["loss_pct"] <= 20
        and not name_suggests_date
    ):
        return False, "Values look numeric"

    _, report = safe_to_datetime(sample)

    success_rate = 1 - report["loss_pct"] / 100

    has_date_symbols = (
        sample.astype(str).str.contains(r"[-/.:]|[A-Za-z]{3}", regex=True).mean()
    )

    if success_rate >= 0.95 and (name_suggests_date or has_date_symbols >= 0.5):

        note = ""

        if report["ambiguous"]:
            note = "; some layouts could be day-first or month-first"

        return True, f"{success_rate:.0%} of sampled values parsed{note}"

    return False, f"Only {success_rate:.0%} of sampled values parsed"


def detect_numeric_candidate(series, column_name):

    if not is_text_column(series):
        return False, "Already numeric or non-text"

    if is_identifier_column(column_name):
        return False, "Identifier-like column; keep as text"

    sample = _sample_values(series)

    if sample.empty:
        return False, "No values available"

    _, report = safe_to_numeric(sample)

    success_rate = 1 - report["loss_pct"] / 100

    if success_rate >= 0.95:
        return True, (
            f"{success_rate:.0%} of sampled values are numeric "
            "(currency symbols and commas are handled)"
        )

    return False, f"Only {success_rate:.0%} of sampled values are numeric"


# ==========================================================
# 6. TEXT QUALITY DETECTION
# ==========================================================

def detect_text_issue(series):

    values = series.dropna().astype(str)

    values = values[~_placeholder_mask(values)]

    if values.empty:
        return False, "No text values"

    stripped = values.str.strip()

    issues = []

    if values.ne(stripped).any():
        issues.append("Extra spaces")

    if stripped.str.casefold().nunique() < stripped.nunique():
        issues.append("Capitalization differences")

    suggestions = suggest_value_mappings(series)

    if suggestions:
        groups = len(set(suggestions.values()))
        issues.append(f"Possible spelling variants ({groups} group(s))")

    if issues:
        return True, ", ".join(issues)

    return False, "No obvious formatting issue"


# ==========================================================
# 7. MISSING VALUE RECOMMENDATIONS
# ==========================================================

def recommend_missing_strategy(series, column_name):

    # "-" and "N/A" count as missing for the recommendation.
    series = normalize_placeholders(series)

    if series.isna().sum() == 0:
        return "Keep missing", "No missing values"

    if series.isna().all():
        return "Keep missing", "Entire column is empty"

    if is_identifier_column(column_name):
        return (
            "Keep missing",
            "Identifier-like column: do not invent values"
        )

    prefix = ""

    # A text column holding numbers is judged by its numeric values.
    if is_text_column(series):

        is_numeric, _ = detect_numeric_candidate(series, column_name)

        if is_numeric:
            series, _ = safe_to_numeric(series)
            prefix = "After numeric conversion: "

    if pd.api.types.is_numeric_dtype(series):

        valid = series.dropna()

        if len(valid) < 2:
            return (
                "Keep missing",
                "Too few valid values to recommend imputation"
            )

        skewness = valid.skew()

        if pd.isna(skewness):
            return (
                "Median",
                prefix + "Skewness unavailable; median is a cautious option"
            )

        if abs(skewness) > 1:
            return (
                "Median",
                prefix + f"High skewness ({skewness:.2f}); extreme values "
                "would distort the mean"
            )

        return (
            "Mean",
            prefix + f"Low or moderate skewness ({skewness:.2f})"
        )

    if pd.api.types.is_datetime64_any_dtype(series):
        return (
            "Keep missing",
            "Date column: filling dates with a statistic is rarely correct"
        )

    top_share = series.value_counts(normalize=True).iloc[0]

    if top_share > 0.5:
        return (
            "Mode",
            f"One value covers {top_share:.0%} of the column"
        )

    return (
        "Unknown",
        "No single value dominates, so the mode would be arbitrary"
    )


# ==========================================================
# 8. CATEGORY MAPPING SUGGESTIONS (spelling variants)
# ==========================================================

def _key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _stem(word):
    """Very small stemmer: groceries/grocerys -> grocery, clothings -> clothing."""

    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"

    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def _words_similar(a, b):

    if a == b:
        return True

    sa, sb = _stem(a), _stem(b)

    if sa == sb:
        return True

    # Short words and different digits are never merged
    # (Grade 1 / Grade 2, Type A / Type B).
    if min(len(sa), len(sb)) < 4:
        return False

    if re.sub(r"\D", "", sa) != re.sub(r"\D", "", sb):
        return False

    if sa[0] != sb[0]:
        return False

    return difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.8


def _phrases_similar(words_a, words_b):
    """Same number of words and every word pair similar.
    'North East' vs 'North West' is NOT similar."""

    return (
        len(words_a) == len(words_b)
        and all(_words_similar(a, b) for a, b in zip(words_a, words_b))
    )


def suggest_value_mappings(series, column_name=None):
    """
    Suggest {original value: standard value}.

    1. Values that differ only in case, spaces or punctuation are one group.
    2. Words that are clear typos or plural forms of each other are merged
       (Furnitur -> Furniture, Grocery -> Groceries).
    3. The most common spelling becomes the standard value.

    Names that are genuinely different (Bangalore / Bengaluru,
    Delhi / New Delhi) are not guessed; the user adds them by hand.
    """

    values = series.dropna().astype(str).str.strip()

    values = values[(values != "") & ~_placeholder_mask(values)]

    if values.empty:
        return {}

    if values.nunique() > min(100, max(30, len(values) * 0.5)):
        return {}

    counts = values.value_counts()

    key_counts = {}
    raw_by_key = {}

    for raw, n in counts.items():

        key = _key(raw)

        if not key:
            continue

        key_counts[key] = key_counts.get(key, 0) + n
        raw_by_key.setdefault(key, []).append(raw)

    heads = []
    head_of = {}

    for key in sorted(key_counts, key=key_counts.get, reverse=True):

        best_raw = max(raw_by_key[key], key=counts.get)
        words = re.findall(r"[a-z0-9]+", best_raw.lower())

        for head_key, head_words in heads:
            if _phrases_similar(words, head_words):
                head_of[key] = head_key
                break
        else:
            heads.append((key, words))
            head_of[key] = key

    def standard_name(key):

        raws = raw_by_key[key]

        proper = [
            r for r in raws
            if not (r.islower() or r.isupper())
        ]

        if proper:
            return max(proper, key=counts.get)

        best = max(raws, key=counts.get)

        # Only all-lowercase text is re-cased. Upper-case values (UPI, ATM,
        # TA/TO) are usually codes, so the most common spelling is kept.
        return best.title() if best.islower() else best

    # members of each final group
    group_members = {}

    for key, raws in raw_by_key.items():
        group_members.setdefault(head_of[key], []).extend(raws)

    suggestions = {}

    for head, members in group_members.items():

        # a value with no other spelling is never changed
        if len(set(members)) < 2:
            continue

        standard = standard_name(head)

        for raw in members:
            if raw != standard:
                suggestions[raw] = standard

    return suggestions


def identify_category_mapping_columns(df):

    results = {}

    for column in df.columns:

        series = df[column]

        if not is_text_column(series):
            continue

        if is_identifier_column(column):
            continue

        values = series.dropna().astype(str).str.strip()
        values = values[(values != "") & ~_placeholder_mask(values)]

        if values.empty:
            continue

        unique_values = sorted(values.unique().tolist())
        unique_count = len(unique_values)

        if unique_count > min(100, max(30, len(values) * 0.5)):
            continue

        suggestions = suggest_value_mappings(series, column)

        if not suggestions:
            continue

        results[column] = {
            "unique_count": unique_count,
            "values": unique_values,
            "suggestions": suggestions,
            "needs_mapping": True
        }

    return results


# ==========================================================
# 9. DATA QUALITY PROFILING
# ==========================================================

def _safe_check(function, fallback, *args):
    """
    Run a detector. If it fails on an unusual column, report that instead of
    crashing the whole page. One odd column must never stop the cleaning tool.
    """

    try:
        return function(*args)
    except Exception as error:
        return fallback[0], f"{fallback[1]} ({type(error).__name__})"


def analyze_quality(df):

    rows = []

    for column in df.columns:

        series = df[column]

        missing = int(series.isna().sum())

        hidden_missing = count_placeholders(series)

        missing_percent = (
            (missing + hidden_missing) / len(df) * 100
            if len(df) else 0
        )

        strategy, reason = _safe_check(
            recommend_missing_strategy,
            ("Keep missing", "Could not analyse this column"),
            series, column
        )

        is_date, date_reason = _safe_check(
            detect_date_candidate,
            (False, "Date check failed"),
            series, column
        )

        is_numeric, numeric_reason = _safe_check(
            detect_numeric_candidate,
            (False, "Numeric check failed"),
            series, column
        )

        if is_text_column(series):
            text_issue, text_reason = _safe_check(
                detect_text_issue,
                (False, "Text check failed"),
                series
            )
        else:
            text_issue = False
            text_reason = "Not a text column"

        rows.append({
            "Column": str(column),
            "Data Type": str(series.dtype),
            "Missing": missing,
            "Hidden Missing": hidden_missing,
            "Missing %": round(missing_percent, 2),
            "Unique Values": int(series.nunique()),
            "Identifier-like": is_identifier_column(column),
            "Missing Recommendation": strategy,
            "Recommendation Reason": reason,
            "Possible Numeric": is_numeric,
            "Numeric Reason": numeric_reason,
            "Possible Date": is_date,
            "Date Reason": date_reason,
            "Text Issue": text_issue,
            "Text Issue Details": text_reason
        })

    return pd.DataFrame(rows)


# ==========================================================
# 10. OUTLIER DETECTION USING IQR
# ==========================================================

def detect_outliers_iqr(df, columns=None):

    if columns is None:

        columns = [
            col
            for col in df.select_dtypes(include="number").columns
            if not is_identifier_column(col)
        ]

    results = []

    for column in columns:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((values < lower) | (values > upper)).sum()
        )

        results.append({
            "Column": str(column),
            "Outlier Count": count,
            "Lower Boundary": float(lower),
            "Upper Boundary": float(upper),
            "Method": "IQR"
        })

    return pd.DataFrame(results)


# ==========================================================
# 11. PREPARATION: PLACEHOLDERS, NUMERIC AND DATE CONVERSION
#     (shared by the page preview and the real cleaning run,
#      so what the user previews is exactly what is applied)
# ==========================================================

def prepare_dataset(
    df,
    numeric_columns=None,
    date_columns=None,
    convert_placeholders=True,
    date_dayfirst=True,
    allow_conversion_loss=False,
    max_conversion_loss_pct=MAX_CONVERSION_LOSS_PCT,
    log=None
):

    prepared = df.copy()

    log = [] if log is None else log

    details = {"numeric": {}, "date": {}}

    numeric_columns = numeric_columns or []
    date_columns = date_columns or []

    # ---- placeholders -> real missing values -------------------

    if convert_placeholders:

        for column in prepared.columns:

            if not is_text_column(prepared[column]):
                continue

            count = count_placeholders(prepared[column])

            if count:

                prepared[column] = normalize_placeholders(
                    prepared[column]
                )

                log.append(
                    f"{column}: Converted {count} placeholder value(s) "
                    "(such as '-' or 'N/A') to missing."
                )

    # ---- numeric conversion -------------------------------------

    for column in numeric_columns:

        if column not in prepared.columns:
            continue

        original = prepared[column]

        if not is_text_column(original):
            log.append(
                f"{column}: Numeric conversion skipped; "
                "the column is already numeric."
            )
            continue

        try:
            converted, report = safe_to_numeric(original)
        except Exception as error:
            log.append(
                f"WARNING: {column}: Numeric conversion SKIPPED "
                f"({type(error).__name__}: {error}). Column left unchanged."
            )
            continue

        details["numeric"][column] = report

        if (
            report["loss_pct"] > max_conversion_loss_pct
            and not allow_conversion_loss
        ):
            log.append(
                f"WARNING: {column}: Numeric conversion SKIPPED. "
                f"{report['failed']} of {report['attempted']} values "
                f"({report['loss_pct']:.1f}%) could not be read, e.g. "
                f"{report['failed_examples']}. The column was left unchanged."
            )
            continue

        prepared[column] = converted

        message = (
            f"{column}: Converted to numeric. "
            f"{report['failed']} value(s) could not be converted"
        )

        if report["failed"]:
            message += f" (e.g. {report['failed_examples']})"

        log.append(message + ".")

    # ---- date conversion ----------------------------------------

    for column in date_columns:

        if column not in prepared.columns:
            continue

        original = prepared[column]

        if pd.api.types.is_datetime64_any_dtype(original):
            continue

        if pd.api.types.is_numeric_dtype(original):
            log.append(
                f"WARNING: {column}: Date conversion SKIPPED. "
                "The column is numeric (numbers such as 20250105 or "
                "Excel serial numbers are not converted automatically)."
            )
            continue

        try:
            converted, report = safe_to_datetime(
                original,
                dayfirst=date_dayfirst
            )
        except Exception as error:
            log.append(
                f"WARNING: {column}: Date conversion SKIPPED "
                f"({type(error).__name__}: {error}). Column left unchanged."
            )
            continue

        details["date"][column] = report

        if (
            report["loss_pct"] > max_conversion_loss_pct
            and not allow_conversion_loss
        ):
            log.append(
                f"WARNING: {column}: Date conversion SKIPPED. "
                f"{report['failed']} of {report['attempted']} values "
                f"({report['loss_pct']:.1f}%) could not be read, e.g. "
                f"{report['failed_examples']}. The column was left unchanged."
            )
            continue

        prepared[column] = converted

        message = (
            f"{column}: Converted to datetime. "
            f"{report['failed']} value(s) could not be converted"
        )

        if report["failed"]:
            message += f" (e.g. {report['failed_examples']})"

        log.append(message + ".")

        if report["ambiguous"]:
            order = "day-first" if date_dayfirst else "month-first"
            log.append(
                f"NOTE: {column}: layouts {report['ambiguous']} could be "
                f"read either way; {order} was used for them."
            )

    return prepared, log, details


# ==========================================================
# 11b. COLUMNS THAT ARE MOSTLY EMPTY
# ==========================================================

def suggest_columns_to_drop(df, missing_threshold=50.0):
    """
    Columns that are probably not worth keeping:
    - at least `missing_threshold` percent missing (placeholders included)
    - or only one distinct value (carries no information)

    This only SUGGESTS. Nothing is removed until the user selects it.
    """

    rows = []

    for column in df.columns:

        series = df[column]

        total = len(series)

        if total == 0:
            continue

        missing = int(series.isna().sum()) + count_placeholders(series)

        missing_pct = missing / total * 100

        if missing_pct >= missing_threshold:

            rows.append({
                "Column": column,
                "Missing": missing,
                "Missing %": round(missing_pct, 1),
                "Reason": f"{missing_pct:.0f}% of the values are missing"
            })

        elif series.dropna().nunique() <= 1 and missing_pct < 100:

            rows.append({
                "Column": column,
                "Missing": missing,
                "Missing %": round(missing_pct, 1),
                "Reason": "Only one distinct value: carries no information"
            })

    return pd.DataFrame(
        rows,
        columns=["Column", "Missing", "Missing %", "Reason"]
    )


# ==========================================================
# 12. APPLY GENERAL NUMERIC VALIDATION
# ==========================================================

def apply_numeric_validation(df, rules, log):

    for column, rule in rules.items():

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            log.append(
                f"{column}: Numeric validation skipped because "
                "the column is not numeric."
            )
            continue

        series = df[column]

        invalid = pd.Series(False, index=df.index)

        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        integer_only = rule.get("integer_only", False)
        action = rule.get("action", "Flag invalid values")

        if minimum is not None:
            invalid |= series.notna() & (series < minimum)

        if maximum is not None:
            invalid |= series.notna() & (series > maximum)

        if integer_only:
            invalid |= series.notna() & (series % 1 != 0)

        count = int(invalid.sum())

        status_column = f"{column}_validation_status"

        if action != "Flag invalid values" and status_column in df.columns:
            df = df.drop(columns=[status_column])

        if count == 0:
            log.append(f"{column}: All non-missing values passed validation.")
            continue

        if action == "Flag invalid values":

            status = pd.Series(
                "Not checked",
                index=df.index,
                dtype="string"
            )

            status.loc[series.notna()] = "Valid"
            status.loc[invalid] = "Invalid"

            df[status_column] = status

            log.append(
                f"{column}: Flagged {count} invalid value(s) in "
                f"'{status_column}'. Rows were retained."
            )

        elif action == "Replace with missing (NA)":

            df[column] = df[column].mask(invalid)

            log.append(
                f"{column}: Replaced {count} invalid value(s) with missing (NA). "
                "Rows were retained."
            )

        elif action == "Drop invalid rows":

            before = len(df)
            df = df.loc[~invalid].copy()
            removed = before - len(df)

            log.append(
                f"{column}: Dropped {removed} row(s) containing values "
                "outside the selected validation rules."
            )

        else:
            log.append(
                f"{column}: Unknown validation action '{action}'. "
                "No changes were made."
            )

    return df


# ==========================================================
# 13. APPLY DATE RELATIONSHIP VALIDATION
# ==========================================================

def apply_date_relationships(df, rules, log):

    for rule in rules:

        earlier = rule.get("earlier")
        later = rule.get("later")
        action = rule.get("action", "Flag only")

        if earlier not in df.columns or later not in df.columns:
            continue

        if not (
            pd.api.types.is_datetime64_any_dtype(df[earlier])
            and pd.api.types.is_datetime64_any_dtype(df[later])
        ):
            continue

        invalid = (
            df[earlier].notna()
            & df[later].notna()
            & (df[later] < df[earlier])
        )

        count = int(invalid.sum())

        if count == 0:
            continue

        if action == "Set later date to missing":

            df.loc[invalid, later] = pd.NaT

            log.append(
                f"{later}: Set {count} date value(s) to missing "
                f"because they occurred before {earlier}."
            )

        elif action == "Set earlier date to missing":

            df.loc[invalid, earlier] = pd.NaT

            log.append(
                f"{earlier}: Set {count} date value(s) to missing "
                f"because they occurred after {later}."
            )

        elif action == "Remove invalid rows":

            df = df.loc[~invalid].copy()

            log.append(
                f"Removed {count} row(s) with invalid date relationships "
                f"between {earlier} and {later}."
            )

        else:

            log.append(
                f"Warning: {count} row(s) have {later} earlier "
                f"than {earlier}. Rows retained for review."
            )

    return df


# ==========================================================
# 14. CATEGORY MAPPING (applies the mapping the user approved)
# ==========================================================

def apply_value_mapping(series, mapping):
    """
    Replace values using the approved mapping.

    Matching is exact first. If there is no exact match, a case-insensitive
    match is used, but only when it points to a single replacement.
    """

    exact = {str(old).strip(): new for old, new in mapping.items()}

    folded = {}

    for old, new in exact.items():
        folded.setdefault(old.casefold(), set()).add(new)

    folded = {
        key: next(iter(targets))
        for key, targets in folded.items()
        if len(targets) == 1
    }

    changed = 0

    def replace_value(value):

        nonlocal changed

        if pd.isna(value):
            return value

        key = str(value).strip()

        if key in exact:
            replacement = exact[key]

        elif key.casefold() in folded:
            replacement = folded[key.casefold()]

        else:
            return value

        if str(value) != str(replacement):
            changed += 1

        return replacement

    return series.apply(replace_value), changed


# ==========================================================
# 15. MAIN CLEANING FUNCTION
#
# Order matters. Statistics (mean, median, mode) must be calculated
# AFTER wrong values, duplicates and outliers have been handled:
#
#   1. placeholders -> missing         ("-" , "N/A")
#   2. numeric / date conversion       (with data-loss guard)
#   3. category mapping + text format  (spelling variants)
#   4. validation rules                (negative quantity ...)
#   5. duplicates                      (key-based, then full rows)
#   6. outliers
#   7. missing values                  (fill last)
# ==========================================================

def clean_dataset(
    df,
    missing_strategies=None,
    numeric_columns=None,
    date_columns=None,
    text_rules=None,
    value_mappings=None,
    numeric_validation_rules=None,
    date_relationship_rules=None,
    duplicate_columns=None,
    duplicate_action="Keep all",
    outlier_columns=None,
    outlier_action="Keep outliers",
    convert_placeholders=True,
    date_dayfirst=True,
    allow_conversion_loss=False,
    missing_custom_values=None,
    add_imputed_flag=False,
    drop_columns=None,
    remove_full_duplicates=True
):

    log = []

    drop_columns = [
        col for col in (drop_columns or []) if col in df.columns
    ]

    # Columns that will be dropped are not converted or validated.
    numeric_columns = [
        col for col in (numeric_columns or []) if col not in drop_columns
    ]
    date_columns = [
        col for col in (date_columns or []) if col not in drop_columns
    ]

    missing_strategies = missing_strategies or {}
    text_rules = text_rules or {}
    value_mappings = value_mappings or {}
    numeric_validation_rules = numeric_validation_rules or {}
    date_relationship_rules = date_relationship_rules or []
    duplicate_columns = duplicate_columns or []
    outlier_columns = outlier_columns or []
    missing_custom_values = missing_custom_values or {}

    # ------------------------------------------------------
    # 1-2. PLACEHOLDERS, NUMERIC AND DATE CONVERSION
    # ------------------------------------------------------

    cleaned, _, _ = prepare_dataset(
        df,
        numeric_columns=numeric_columns,
        date_columns=date_columns,
        convert_placeholders=convert_placeholders,
        date_dayfirst=date_dayfirst,
        allow_conversion_loss=allow_conversion_loss,
        log=log
    )

    # ------------------------------------------------------
    # 2b. DROP SELECTED COLUMNS
    #     (after placeholders were converted, so the reported missing
    #      percentage counts '-' and 'N/A' as well)
    # ------------------------------------------------------

    for column in drop_columns:

        missing_pct = cleaned[column].isna().mean() * 100

        log.append(
            f"{column}: Dropped the whole column "
            f"({missing_pct:.1f}% missing)."
        )

    if drop_columns:
        cleaned = cleaned.drop(columns=drop_columns)

    # ------------------------------------------------------
    # 3a. CUSTOM CATEGORY MAPPINGS
    # ------------------------------------------------------

    for column, mapping in value_mappings.items():

        if column not in cleaned.columns or not mapping:
            continue

        cleaned[column], changed = apply_value_mapping(
            cleaned[column],
            mapping
        )

        log.append(
            f"{column}: Applied custom mappings to "
            f"{changed} value(s)."
        )

    # ------------------------------------------------------
    # 3b. SAFE TEXT STANDARDIZATION
    # ------------------------------------------------------

    for column, rule in text_rules.items():

        if column not in cleaned.columns:
            continue

        if not is_text_column(cleaned[column]):
            continue

        before = cleaned[column].copy()

        series = cleaned[column].astype("string")

        if rule.get("trim", True):
            series = series.str.strip()

        case_option = rule.get("case", "Preserve")

        if case_option == "lowercase":
            series = series.str.lower()

        elif case_option == "UPPERCASE":
            series = series.str.upper()

        elif case_option == "Title Case":
            series = series.str.title()

        cleaned[column] = series

        # compare as plain strings: works for object, string and
        # categorical columns (a categorical cannot be filled with "")
        changed = int(
            (
                before.astype("string").fillna("")
                != cleaned[column].astype("string").fillna("")
            ).sum()
        )

        if changed:
            log.append(
                f"{column}: Applied text formatting. "
                f"{changed} value(s) changed."
            )

    # ------------------------------------------------------
    # 4. VALIDATION RULES
    # ------------------------------------------------------

    cleaned = apply_numeric_validation(
        cleaned,
        numeric_validation_rules,
        log
    )

    cleaned = apply_date_relationships(
        cleaned,
        date_relationship_rules,
        log
    )

    # ------------------------------------------------------
    # 5a. KEY-BASED DUPLICATES
    # ------------------------------------------------------

    if duplicate_columns:

        valid_keys = [
            col for col in duplicate_columns
            if col in cleaned.columns
        ]

        if valid_keys and duplicate_action != "Keep all":

            before = len(cleaned)

            group_rows = int(
                cleaned.duplicated(subset=valid_keys, keep=False).sum()
            )

            if group_rows:

                if duplicate_action == "Keep first":

                    cleaned = cleaned.drop_duplicates(
                        subset=valid_keys,
                        keep="first"
                    )

                elif duplicate_action == "Keep last":

                    cleaned = cleaned.drop_duplicates(
                        subset=valid_keys,
                        keep="last"
                    )

                elif duplicate_action == "Keep most complete":

                    cleaned["_completeness_score"] = (
                        cleaned.notna().sum(axis=1)
                    )

                    cleaned = (
                        cleaned.sort_values(
                            "_completeness_score",
                            ascending=False,
                            kind="stable"
                        )
                        .drop_duplicates(
                            subset=valid_keys,
                            keep="first"
                        )
                        .drop(columns="_completeness_score")
                        .sort_index()
                    )

                log.append(
                    f"Duplicate key check on {valid_keys}: "
                    f"{group_rows} row(s) belonged to duplicate groups; "
                    f"removed {before - len(cleaned)} row(s) "
                    f"({duplicate_action})."
                )

            else:

                log.append(
                    f"No duplicate key values found for {valid_keys}."
                )

        elif valid_keys:

            log.append(
                f"Duplicate key check on {valid_keys}: "
                "action is 'Keep all', so no rows were removed."
            )

    # ------------------------------------------------------
    # 5b. FULL ROW DUPLICATES
    #     (after text standardization, so rows that only differed
    #      by spelling or spacing are now recognised as duplicates)
    # ------------------------------------------------------

    full_duplicates = int(cleaned.duplicated().sum())

    if full_duplicates and remove_full_duplicates:

        cleaned = cleaned.drop_duplicates()

        log.append(
            f"Removed {full_duplicates} completely identical row(s)."
        )

    elif full_duplicates:

        log.append(
            f"NOTE: {full_duplicates} completely identical row(s) were "
            "kept because duplicate removal is switched off."
        )

    # ------------------------------------------------------
    # 6. OUTLIER HANDLING (before filling missing values, so the
    #    mean/median is not distorted by extreme values)
    # ------------------------------------------------------

    if outlier_action != "Keep outliers":

        for column in outlier_columns:

            if column not in cleaned.columns:
                continue

            if not pd.api.types.is_numeric_dtype(cleaned[column]):
                continue

            values = cleaned[column].dropna()

            if len(values) < 4:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            mask = (
                (cleaned[column] < lower)
                | (cleaned[column] > upper)
            )

            count = int(mask.sum())

            if count == 0:
                continue

            if outlier_action == "Cap outliers":

                cleaned[column] = cleaned[column].clip(
                    lower=lower,
                    upper=upper
                )

                log.append(
                    f"{column}: Capped {count} potential outlier(s) "
                    f"to [{lower:,.2f}, {upper:,.2f}]."
                )

            elif outlier_action == "Remove outlier rows":

                cleaned = cleaned.loc[~mask].copy()

                log.append(
                    f"{column}: Removed {count} row(s) "
                    "with potential outliers."
                )

    # ------------------------------------------------------
    # 7. MISSING VALUE HANDLING (last)
    # ------------------------------------------------------

    for column in list(cleaned.columns):

        strategy = missing_strategies.get(
            column, "Keep missing"
        )

        if strategy == "Keep missing":
            continue

        missing_mask = cleaned[column].isna()

        count = int(missing_mask.sum())

        if count == 0:
            continue

        if missing_mask.all():
            log.append(
                f"{column}: Imputation skipped because "
                "the entire column is empty."
            )
            continue

        if strategy == "Drop rows":

            cleaned = cleaned.loc[~missing_mask].copy()

            log.append(
                f"{column}: Removed {count} row(s) with missing values."
            )

            continue

        is_numeric = pd.api.types.is_numeric_dtype(cleaned[column])

        if strategy in ("Mean", "Median"):

            if not is_numeric:
                log.append(
                    f"{column}: {strategy} unavailable for non-numeric data. "
                    "Convert the column to numeric first."
                )
                continue

            fill_value = (
                cleaned[column].mean()
                if strategy == "Mean"
                else cleaned[column].median()
            )

        elif strategy == "Mode":

            modes = cleaned[column].mode(dropna=True)

            if modes.empty:
                continue

            fill_value = modes.iloc[0]

        elif strategy == "Unknown":

            if is_numeric or pd.api.types.is_datetime64_any_dtype(
                cleaned[column]
            ):
                log.append(
                    f"{column}: 'Unknown' can only be used for text columns."
                )
                continue

            fill_value = "Unknown"

        elif strategy == "Custom value":

            raw_value = str(missing_custom_values.get(column, "")).strip()

            if raw_value == "":
                log.append(
                    f"{column}: Custom value not supplied; "
                    "missing values retained."
                )
                continue

            if is_numeric:
                try:
                    fill_value = float(raw_value)
                except ValueError:
                    log.append(
                        f"{column}: '{raw_value}' is not a number; "
                        "missing values retained."
                    )
                    continue

            elif pd.api.types.is_datetime64_any_dtype(cleaned[column]):
                fill_value = pd.to_datetime(raw_value, errors="coerce")

                if pd.isna(fill_value):
                    log.append(
                        f"{column}: '{raw_value}' is not a valid date; "
                        "missing values retained."
                    )
                    continue

            else:
                fill_value = raw_value

        else:
            continue

        if add_imputed_flag:

            flag_name = f"{column}_was_missing"

            if flag_name not in cleaned.columns:
                cleaned[flag_name] = missing_mask

        if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
            cleaned[column] = cleaned[column].astype("string")

        cleaned[column] = cleaned[column].fillna(fill_value)

        log.append(
            f"{column}: Filled {count} missing value(s) "
            f"using {strategy}."
        )

    if not log:
        log.append("No cleaning operations were applied.")

    return cleaned, log


# ==========================================================
# 16. FINAL QUALITY SUMMARY AND SAFETY WARNINGS
# ==========================================================

def final_quality_summary(original, cleaned):

    original_rows = len(original)

    removed = original_rows - len(cleaned)

    return {
        "Original Rows": original_rows,
        "Cleaned Rows": len(cleaned),
        "Rows Removed": removed,
        "Rows Removed %": (
            round(removed / original_rows * 100, 2)
            if original_rows else 0.0
        ),
        "Original Missing": int(original.isna().sum().sum()),
        "Remaining Missing": int(cleaned.isna().sum().sum()),
        "Original Duplicates": int(original.duplicated().sum()),
        "Remaining Duplicates": int(cleaned.duplicated().sum()),
        "Original Columns": len(original.columns),
        "Cleaned Columns": len(cleaned.columns)
    }


def quality_warnings(original, cleaned, max_row_loss_pct=10.0,
                     max_new_missing_pct=5.0):
    """
    Plain-language warnings shown after cleaning, so that data loss
    is never silent.
    """

    warnings = []

    if len(original) == 0:
        return warnings

    removed = len(original) - len(cleaned)
    removed_pct = removed / len(original) * 100

    if removed_pct > max_row_loss_pct:
        warnings.append(
            f"{removed:,} rows ({removed_pct:.1f}%) were removed. "
            "Check the duplicate, validation, outlier and missing-value "
            "settings."
        )

    for column in original.columns:

        if column not in cleaned.columns or len(cleaned) == 0:
            continue

        before = int(original[column].isna().sum())
        after = int(cleaned[column].isna().sum())

        increase_pct = (after - before) / len(cleaned) * 100

        if increase_pct > max_new_missing_pct:
            warnings.append(
                f"'{column}': missing values rose from {before:,} to "
                f"{after:,}. Values may have been blanked by a conversion "
                "or validation rule. Check the cleaning log."
            )

    for column in cleaned.columns:

        if len(cleaned) and cleaned[column].isna().mean() > 0.5:
            warnings.append(
                f"'{column}' is more than 50% missing after cleaning."
            )

    return warnings