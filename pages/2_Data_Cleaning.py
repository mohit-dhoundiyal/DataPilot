import streamlit as st
import pandas as pd
import plotly.express as px

from io import BytesIO

from engine.cleaner import (
    analyze_quality,
    detect_outliers_iqr,
    detect_numeric_candidate,
    detect_date_candidate,
    is_identifier_column,
    is_text_column,
    identify_category_mapping_columns,
    prepare_dataset,
    clean_dataset,
    final_quality_summary,
    quality_warnings,
    suggest_columns_to_drop
)

from utils.ui_styles import inject_css
inject_css()


DAY_FIRST_LABEL = "Day first (31/12/2025 = 31 December)"
MONTH_FIRST_LABEL = "Month first (12/31/2025 = 31 December)"


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.title("DataPilot | Data Cleaning")

st.caption(
    "Detect data quality issues, configure validation rules, "
    "review recommendations and clean your dataset safely."
)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

uploaded_df = st.session_state.get("df")

if uploaded_df is None:

    st.warning(
        "Upload a CSV or Excel dataset from the Home page first."
    )

    st.stop()

if uploaded_df.empty or len(uploaded_df.columns) == 0:

    st.warning("The uploaded dataset is empty.")

    st.stop()

if not uploaded_df.columns.is_unique:

    st.error(
        "Your dataset contains duplicate column names. "
        "Rename them before cleaning."
    )

    st.stop()


# Detect a new upload using the original uploaded dataset.
# Only the first and last 500 rows are hashed, so large files stay fast.
try:

    hash_sample = pd.concat(
        [uploaded_df.head(500), uploaded_df.tail(500)]
    )

    dataset_signature = (
        uploaded_df.shape,
        tuple(map(str, uploaded_df.columns)),
        tuple(map(str, uploaded_df.dtypes)),
        int(
            pd.util.hash_pandas_object(
                hash_sample.astype(str),
                index=True
            ).sum()
        )
    )

except Exception:

    dataset_signature = (
        uploaded_df.shape,
        tuple(map(str, uploaded_df.columns))
    )


if (
    st.session_state.get("cleaning_dataset_signature")
    != dataset_signature
):

    st.session_state["cleaning_dataset_signature"] = dataset_signature

    # A new upload starts a fresh cleaning workflow.
    st.session_state["cleaned_df"] = None
    st.session_state["cleaning_log"] = None
    st.session_state["cleaning_round_source_df"] = None
    st.session_state["cleaning_round_number"] = 0
    st.session_state["cleaning_applied"] = False
    st.session_state["value_mappings"] = {}
    st.session_state["date_relationship_rules"] = []
    st.session_state["_reset_cleaning_controls"] = True


# After applying a cleaning round, clear operation selections before
# showing the latest cleaned data as the next round's input.
#
# IMPORTANT: every setting that can change data must be reset here,
# including the duplicate and outlier ACTIONS. Otherwise an action such
# as "Remove outlier rows" would silently run again in the next round.
if st.session_state.pop("_reset_cleaning_controls", False):

    reset_keys = [
        "numeric_columns",
        "date_columns",
        "identifier_text_columns_widget",
        "validation_numeric_columns",
        "duplicate_key_columns",
        "duplicate_action",
        "outlier_columns",
        "outlier_action",
        "missing_strategies",
        "missing_custom_values",
        "numeric_validation_rules",
        "text_rules",
        "relationship_earlier",
        "relationship_later",
        "date_relationship_action",
        "convert_placeholders",
        "date_dayfirst_choice",
        "allow_conversion_loss",
        "add_imputed_flag",
        "category_mapping_column",
        "drop_columns",
        "drop_threshold",
        "remove_full_duplicates",
    ]

    for key in reset_keys:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if (
            key.startswith("min_enabled_")
            or key.startswith("min_value_")
            or key.startswith("max_enabled_")
            or key.startswith("max_value_")
            or key.startswith("integer_")
            or key.startswith("validation_action_")
            or key.startswith("missing_")
            or key.startswith("trim_")
            or key.startswith("case_")
            or key.startswith("category_mapping_")
            or key.startswith("save_mapping_")
        ):
            st.session_state.pop(key, None)

    st.session_state["value_mappings"] = {}
    st.session_state["date_relationship_rules"] = []


# Use the latest cleaned dataset as the next round's input.
latest_cleaned_df = st.session_state.get("cleaned_df")

if (
    st.session_state.get("cleaning_applied", False)
    and latest_cleaned_df is not None
):
    df = latest_cleaned_df.copy()
    current_source_label = "Latest cleaned dataset"
else:
    df = uploaded_df.copy()
    current_source_label = "Original uploaded dataset"


if df.empty or len(df.columns) == 0:

    st.warning("The current dataset is empty. Review the previous cleaning operations.")

    st.stop()


# ==========================================================
# 3. INITIAL DATA QUALITY ANALYSIS
# ==========================================================

quality = analyze_quality(df)

total_rows = len(df)

total_columns = len(df.columns)

total_missing = int(df.isna().sum().sum())

total_hidden_missing = int(quality["Hidden Missing"].sum())

total_duplicates = int(df.duplicated().sum())


# ==========================================================
# 4. DATASET OVERVIEW
# ==========================================================

st.subheader("Dataset overview")

st.caption(f"Current cleaning input: {current_source_label}")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Total rows",
    f"{total_rows:,}"
)

m2.metric(
    "Total columns",
    f"{total_columns:,}"
)

m3.metric(
    "Missing values",
    f"{total_missing + total_hidden_missing:,}",
    help=(
        "Empty cells plus placeholders such as '-', 'N/A' or 'null' "
        "that really mean 'no value'."
    )
)

m4.metric(
    "Full-row duplicates",
    f"{total_duplicates:,}"
)

if total_hidden_missing:

    st.caption(
        f"{total_missing:,} empty cells and {total_hidden_missing:,} "
        "placeholder values (for example '-' or 'N/A') are counted as missing."
    )

st.divider()


# ==========================================================
# 5. MAIN TABS
#
# The tabs follow the order in which the cleaning is performed.
# "Data Types & Text" comes first because every later tab previews
# the data AFTER those conversions (a text column holding prices can
# only be checked for outliers or filled with a median once it is numeric).
# ==========================================================

(
    tab_overview,
    tab_types,
    tab_validation,
    tab_missing,
    tab_duplicates,
    tab_apply
) = st.tabs([
    "Quality Overview",
    "Data Types & Text",
    "Validation Rules",
    "Missing Values",
    "Duplicates & Outliers",
    "Apply & Results"
])


# ==========================================================
# TAB 1: QUALITY OVERVIEW
# ==========================================================

with tab_overview:

    st.subheader("Data quality report")

    st.write(
        "DataPilot examines each column and identifies "
        "potential data quality issues."
    )

    overview_columns = [
        "Column",
        "Data Type",
        "Missing",
        "Hidden Missing",
        "Missing %",
        "Unique Values",
        "Identifier-like",
        "Missing Recommendation",
        "Possible Numeric",
        "Possible Date",
        "Text Issue",
        "Text Issue Details"
    ]

    st.dataframe(
        quality[overview_columns],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Missing values by column")

    missing_data = quality.assign(
        **{"Missing (incl. placeholders)": (
            quality["Missing"] + quality["Hidden Missing"]
        )}
    )

    missing_data = missing_data[
        missing_data["Missing (incl. placeholders)"] > 0
    ][["Column", "Missing (incl. placeholders)"]]

    if not missing_data.empty:

        fig = px.bar(
            missing_data,
            x="Column",
            y="Missing (incl. placeholders)",
            text="Missing (incl. placeholders)",
            title="Missing values in each column"
        )

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.success("No missing values detected.")

    st.divider()

    st.subheader("Data quality indicators")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Columns with missing values",
        int(((quality["Missing"] + quality["Hidden Missing"]) > 0).sum())
    )

    c2.metric(
        "Possible numeric columns",
        int(quality["Possible Numeric"].sum())
    )

    c3.metric(
        "Possible date columns",
        int(quality["Possible Date"].sum())
    )

    c4.metric(
        "Text columns with issues",
        int(quality["Text Issue"].sum())
    )

    with st.expander("Preview original dataset"):

        st.dataframe(
            df.head(20),
            use_container_width=True
        )


# ==========================================================
# TAB 2: DATA TYPES AND TEXT STANDARDIZATION
# ==========================================================

with tab_types:

    st.subheader("Data type detection")

    st.write(
        "DataPilot suggests conversions based on column values. "
        "Always review the suggestions and the preview before applying them."
    )

    # ------------------------------------------------------
    # PLACEHOLDERS
    # ------------------------------------------------------

    st.markdown("### Hidden missing values")

    convert_placeholders = st.checkbox(
        "Treat placeholders such as '-', 'N/A', 'null' and 'none' "
        "as missing values",
        value=True,
        key="convert_placeholders"
    )

    placeholder_columns = quality[quality["Hidden Missing"] > 0]

    if placeholder_columns.empty:

        st.caption("No placeholder values were found.")

    else:

        st.info(
            "Placeholders found: "
            + ", ".join(
                f"{row['Column']} ({int(row['Hidden Missing'])})"
                for _, row in placeholder_columns.iterrows()
            )
        )

        st.caption(
            "Turn this off if a value such as 'None' is a real category "
            "in your data."
        )

    st.divider()

    # ------------------------------------------------------
    # NUMERIC CONVERSION
    # ------------------------------------------------------

    st.markdown("### Numeric conversion")

    text_columns = [
        col for col in df.columns
        if is_text_column(df[col])
    ]

    numeric_suggestions = []

    for column in text_columns:

        detected, reason = detect_numeric_candidate(
            df[column],
            column
        )

        if detected:
            numeric_suggestions.append(column)

    if numeric_suggestions:

        st.info(
            "Potential numeric columns: "
            + ", ".join(map(str, numeric_suggestions))
            + ". Currency symbols (Rs., ₹, $) and thousands "
            "separators are handled automatically."
        )

    numeric_columns = st.multiselect(
        "Select columns to convert to numeric",
        options=text_columns,
        default=numeric_suggestions,
        key="numeric_columns"
    )

    st.divider()

    # ------------------------------------------------------
    # DATE CONVERSION
    # ------------------------------------------------------

    st.markdown("### Date and time conversion")

    date_suggestions = []

    for column in df.columns:

        detected, reason = detect_date_candidate(
            df[column],
            column
        )

        if detected and not pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):

            date_suggestions.append(column)

    if date_suggestions:

        st.info(
            "Potential date columns: "
            + ", ".join(map(str, date_suggestions))
        )

    date_columns = st.multiselect(
        "Select columns to convert to datetime",
        options=list(df.columns),
        default=date_suggestions,
        key="date_columns"
    )

    date_order = st.radio(
        "If a date could be read either way (for example 04/05/2025), use:",
        [DAY_FIRST_LABEL, MONTH_FIRST_LABEL],
        key="date_dayfirst_choice",
        horizontal=True
    )

    st.caption(
        "Columns that mix several date layouts are handled layout by "
        "layout. A layout such as 25/10/2025 can only be day-first, so "
        "this choice only affects layouts where every value is ambiguous."
    )

    st.divider()

    # ------------------------------------------------------
    # DATA-LOSS PROTECTION
    # ------------------------------------------------------

    allow_conversion_loss = st.checkbox(
        "Allow conversions that turn more than 5% of values into "
        "missing values (not recommended)",
        value=False,
        key="allow_conversion_loss"
    )

    st.caption(
        "By default a conversion that would lose more than 5% of the "
        "values is skipped and the column is left unchanged."
    )

    # The data used by every later tab: placeholders handled and the
    # selected conversions applied. It is exactly what the cleaning run
    # will start from, so the previews below match the final result.
    prepared_df, prepare_log, prepare_details = prepare_dataset(
        df,
        numeric_columns=numeric_columns,
        date_columns=date_columns,
        convert_placeholders=convert_placeholders,
        date_dayfirst=(date_order == DAY_FIRST_LABEL),
        allow_conversion_loss=allow_conversion_loss
    )

    if numeric_columns or date_columns:

        with st.expander(
            "Preview of the selected conversions",
            expanded=True
        ):

            for column, report in prepare_details["numeric"].items():

                if report["failed"]:
                    st.write(
                        f"**{column}**: {report['failed']} of "
                        f"{report['attempted']} values "
                        f"({report['loss_pct']:.1f}%) cannot be read as "
                        f"numbers, e.g. {report['failed_examples']}"
                    )
                else:
                    st.write(
                        f"**{column}**: all {report['attempted']:,} values "
                        "can be read as numbers."
                    )

            for column, report in prepare_details["date"].items():

                st.write(
                    f"**{column}**: {report['failed']} of "
                    f"{report['attempted']} values "
                    f"({report['loss_pct']:.1f}%) cannot be read as dates."
                )

                if report["groups"]:

                    group_table = pd.DataFrame([
                        {
                            "Layout": layout,
                            "Format used": info["format"],
                            "Rows": info["rows"],
                            "Unreadable": info["failed"],
                            "Ambiguous": "Yes" if info["ambiguous"] else "No"
                        }
                        for layout, info in report["groups"].items()
                    ])

                    st.dataframe(
                        group_table,
                        use_container_width=True,
                        hide_index=True
                    )

            for line in prepare_log:

                if line.startswith("WARNING"):
                    st.error(line)

    st.divider()

    # ------------------------------------------------------
    # IDENTIFIER AND PHONE PROTECTION
    # ------------------------------------------------------

    st.markdown("### Identifier and contact columns")

    identifier_columns = [
        col for col in df.columns
        if is_identifier_column(col)
    ]

    if identifier_columns:

        st.write(
            "These columns appear to contain identifiers or contact details:"
        )

        st.write(", ".join(map(str, identifier_columns)))

        st.info(
            "These columns are excluded from automatic numeric conversion "
            "and default outlier detection. Store identifiers and phone "
            "numbers as text, not numerical measurements."
        )

        numeric_identifier_columns = [
            col for col in identifier_columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        if numeric_identifier_columns:

            st.warning(
                "Some identifier columns are currently numeric. "
                "You can convert them to text below. "
                "Lost leading zeros cannot be recovered automatically."
            )

            identifier_text_columns = st.multiselect(
                "Convert selected identifier columns to text",
                numeric_identifier_columns,
                key="identifier_text_columns_widget")

        else:

            identifier_text_columns = []

    else:

        st.caption(
            "No obvious identifier or contact columns were detected."
        )

        identifier_text_columns = []

    st.session_state["identifier_text_columns"] = (
        identifier_text_columns
    )

    st.divider()

    # ------------------------------------------------------
    # TEXT FORMATTING
    # ------------------------------------------------------

    st.markdown("### Text formatting")

    st.write(
        "Select how text values should be formatted. "
        "You can preserve the original capitalization if required."
    )

    # Columns that are still text after the selected conversions.
    formatting_columns = [
        col for col in prepared_df.columns
        if is_text_column(prepared_df[col])
    ]

    text_rules = {}

    for column in formatting_columns:

        with st.expander(f"Text formatting: {column}"):

            trim = st.checkbox(
                "Remove leading and trailing spaces",
                value=True,
                key=f"trim_{column}"
            )

            case_option = st.selectbox(
                "Capitalization",
                [
                    "Preserve",
                    "lowercase",
                    "UPPERCASE",
                    "Title Case"
                ],
                key=f"case_{column}"
            )

            if trim or case_option != "Preserve":

                text_rules[column] = {
                    "trim": trim,
                    "case": case_option
                }

    st.session_state["text_rules"] = text_rules

    st.divider()

    # ------------------------------------------------------
    # CATEGORY VALUE MAPPING
    # ------------------------------------------------------

    st.markdown("### Category Value Mapping")

    st.caption(
        "Suggestions merge spelling variants of the same value "
        "(for example Furnitur / Furniture). Names that are genuinely "
        "different, such as Bangalore and Bengaluru or Delhi and "
        "New Delhi, are never guessed: add them yourself below. "
        "Mappings are applied only after you save them."
    )

    category_candidates = identify_category_mapping_columns(prepared_df)

    if "value_mappings" not in st.session_state:
        st.session_state["value_mappings"] = {}

    if not category_candidates:

        st.success("No category columns requiring value mapping detected.")

    # ------------------------------------------------------
    else:
        st.info(
            f"{len(category_candidates)} categorical column(s) "
            "have inconsistent values."
        )

        if st.button(
            "Accept all suggested mappings",
            key="accept_all_mappings"
        ):

            for candidate_column, candidate in category_candidates.items():
                st.session_state["value_mappings"][
                    candidate_column
                ] = dict(candidate["suggestions"])

            st.success(
                "Suggested mappings saved for all listed columns. "
                "You can still edit any column below."
            )

        selected_column = st.selectbox(
            "Select column for value mapping",
            options=list(category_candidates.keys()),
            key="category_mapping_column"
        )
        details = category_candidates[selected_column]
        st.write("**Existing unique values:**")
        st.write(", ".join(details["values"]))
        st.write("**Suggested mappings:**")
        saved_mappings = st.session_state["value_mappings"].get(
            selected_column,
            details["suggestions"]
        )
        mapping_text = "\n".join(
            f"{old}={new}"
            for old, new in saved_mappings.items()
        )
        edited_mapping = st.text_area(
            "Edit or add mappings (original=replacement, one per line)",
            value=mapping_text,
            key=f"category_mapping_{selected_column}"
        )
        if st.button(
            "Save mapping",
            key=f"save_mapping_{selected_column}"
        ):
            parsed_mapping = {}
            for line in edited_mapping.splitlines():
                if "=" not in line:
                    continue
                old, new = line.split("=", 1)
                old = old.strip()
                new = new.strip()
                if old and new:
                    parsed_mapping[old] = new
            st.session_state["value_mappings"][
                selected_column
            ] = parsed_mapping
            st.success(
                f"Mappings saved for {selected_column}."
            )

    saved_columns = {
        column: len(mapping)
        for column, mapping in st.session_state["value_mappings"].items()
        if mapping
    }

    if saved_columns:

        st.success(
            "Saved mappings: "
            + ", ".join(
                f"{column} ({count} value(s))"
                for column, count in saved_columns.items()
            )
        )

    elif category_candidates:

        st.warning(
            "No mappings are saved yet. Inconsistent values will stay "
            "as they are unless you save a mapping."
        )


# ==========================================================
# TAB 3: CONFIGURABLE VALIDATION RULES
# ==========================================================

with tab_validation:

    st.subheader("Custom data validation rules")

    st.write(
        "Define rules based on what the columns mean in your dataset. "
        "DataPilot will not assume that a value is invalid just because "
        "it is unusually high or low."
    )

    st.caption(
        "Columns selected for numeric conversion on the previous tab "
        "are already shown here as numbers."
    )

    # ------------------------------------------------------
    # NUMERIC RULES
    # ------------------------------------------------------

    st.markdown("### Numeric validation")

    st.caption(
        "Choose whether to flag invalid values, replace only those cells "
        "with missing values, or remove the entire rows. Missing values "
        "created here can then be filled on the Missing Values tab."
    )


    numeric_measurement_columns = [
        col for col in prepared_df.select_dtypes(include="number").columns
        if not is_identifier_column(col)
    ]

    numeric_rules = {}

    if not numeric_measurement_columns:

        st.caption(
            "No numeric measurement columns are available."
        )

    else:

        selected_rule_columns = st.multiselect(
            "Select columns that need numeric validation",
            numeric_measurement_columns,
            key="validation_numeric_columns"
        )

        for index, column in enumerate(selected_rule_columns):

            with st.expander(f"Validation rules: {column}"):

                use_min = st.checkbox(
                    "Set minimum allowed value",
                    key=f"min_enabled_{column}"
                )

                minimum = None

                if use_min:

                    min_text = st.text_input(
                        "Minimum value",
                        key=f"min_value_{column}"
                    )

                    try:
                        minimum = float(min_text)
                    except ValueError:
                        st.warning("Enter a valid minimum number.")

                use_max = st.checkbox(
                    "Set maximum allowed value",
                    key=f"max_enabled_{column}"
                )

                maximum = None

                if use_max:

                    max_text = st.text_input(
                        "Maximum value",
                        key=f"max_value_{column}"
                    )

                    try:
                        maximum = float(max_text)
                    except ValueError:
                        st.warning("Enter a valid maximum number.")

                integer_only = st.checkbox(
                    "Values must be whole numbers",
                    key=f"integer_{column}"
                )

                validation_action = st.selectbox(
                    "Action for values outside the validation rules",
                    [
                        "Flag invalid values",
                        "Replace with missing (NA)",
                        "Drop invalid rows"
                    ],
                    key=f"validation_action_{column}"
                )

                if (
                    minimum is not None
                    or maximum is not None
                    or integer_only
                ):

                    numeric_rules[column] = {
                        "minimum": minimum,
                        "maximum": maximum,
                        "integer_only": integer_only,
                        "action": validation_action
                    }

                    preview_invalid = pd.Series(
                        False,
                        index=prepared_df.index
                    )

                    values = prepared_df[column]

                    if minimum is not None:
                        preview_invalid |= (
                            values.notna() & (values < minimum)
                        )

                    if maximum is not None:
                        preview_invalid |= (
                            values.notna() & (values > maximum)
                        )

                    if integer_only:
                        preview_invalid |= (
                            values.notna() & (values % 1 != 0)
                        )

                    st.info(
                        f"Preview: {int(preview_invalid.sum())} "
                        "invalid value(s) will be affected by this rule."
                    )

    st.session_state["numeric_validation_rules"] = (
        numeric_rules
    )

    st.divider()

    # ------------------------------------------------------
    # DATE RELATIONSHIP RULES
    # ------------------------------------------------------

    st.markdown("### Date relationship validation")

    date_type_columns = [
        col for col in prepared_df.columns
        if pd.api.types.is_datetime64_any_dtype(prepared_df[col])
    ]

    date_relationship_rules = []

    if len(date_type_columns) >= 2:

        st.write(
            "Define relationships between two date columns. "
            "For example, an end date should not precede a start date."
        )

        earlier_column = st.selectbox(
            "Earlier/event start date column",
            date_type_columns,
            key="relationship_earlier"
        )

        later_options = [
            col for col in date_type_columns
            if col != earlier_column
        ]

        later_column = st.selectbox(
            "Later/event end date column",
            later_options,
            key="relationship_later"
        )

        date_action = st.selectbox(
            "Action for rows violating this relationship",
            [
                "Flag only",
                "Set later date to missing",
                "Set earlier date to missing",
                "Remove invalid rows"
            ],
            key="date_relationship_action"
        )

        if st.button(
            "Add date relationship rule",
            key="add_date_rule"
        ):

            rule = {
                "earlier": earlier_column,
                "later": later_column,
                "action": date_action
            }

            existing = st.session_state.get(
                "date_relationship_rules", []
            )

            if rule not in existing:
                existing.append(rule)

            st.session_state[
                "date_relationship_rules"
            ] = existing

            st.success("Date relationship rule added.")

        saved_date_rules = st.session_state.get(
            "date_relationship_rules", []
        )

        if saved_date_rules:

            st.write("Configured date rules:")

            for index, rule in enumerate(saved_date_rules):

                st.write(
                    f"{rule['later']} must not be earlier than "
                    f"{rule['earlier']} — {rule['action']}"
                )

                if st.button(
                    "Remove rule",
                    key=f"remove_date_rule_{index}"
                ):

                    saved_date_rules.pop(index)

                    st.session_state[
                        "date_relationship_rules"
                    ] = saved_date_rules

                    st.rerun()

            date_relationship_rules = saved_date_rules

    else:

        st.caption(
            "Convert at least two columns to datetime before defining "
            "date relationship rules."
        )

    st.session_state["date_relationship_rules"] = (
        date_relationship_rules
    )


# ==========================================================
# TAB 4: MISSING VALUE HANDLING
# ==========================================================

with tab_missing:

    st.subheader("Missing value recommendations")

    st.write(
        "Review the recommended filling method for each column. "
        "You can choose a different method."
    )

    st.info(
        "Default behaviour: missing values are preserved. "
        "Automatic recommendations do not fill values without your approval. "
        "Filling is done last, after duplicates, invalid values and "
        "outliers have been handled, so those do not distort the result."
    )

    st.caption(
        "This tab shows the data after the conversions selected on the "
        "Data Types & Text tab."
    )

    # ------------------------------------------------------
    # DROP COLUMNS THAT ARE MOSTLY EMPTY
    # ------------------------------------------------------

    st.markdown("### Columns with too many missing values")

    st.write(
        "A column that is mostly empty usually cannot be filled "
        "reliably. Filling 90% of a column with one value invents "
        "most of its data, so dropping the column is often better."
    )

    drop_threshold = st.slider(
        "Suggest dropping columns that are at least this % missing",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
        key="drop_threshold"
    )

    drop_suggestions = suggest_columns_to_drop(
        prepared_df,
        drop_threshold
    )

    if drop_suggestions.empty:

        st.caption("No columns reach this threshold.")

    else:

        st.dataframe(
            drop_suggestions,
            use_container_width=True,
            hide_index=True
        )

        def select_suggested_columns():
            st.session_state["drop_columns"] = (
                drop_suggestions["Column"].tolist()
            )

        st.button(
            "Select all suggested columns",
            on_click=select_suggested_columns,
            key="select_suggested_drop"
        )

    drop_columns = st.multiselect(
        "Columns to drop completely",
        options=list(prepared_df.columns),
        key="drop_columns"
    )

    if drop_columns:

        st.warning(
            "These columns will be removed from the cleaned dataset: "
            + ", ".join(map(str, drop_columns))
            + ". Check that none of them is needed for your analysis."
        )

    st.divider()

    prepared_quality = analyze_quality(prepared_df)

    # Columns that will be dropped do not need a filling method.
    missing_columns = prepared_quality[
        (prepared_quality["Missing"] > 0)
        & ~prepared_quality["Column"].isin([str(c) for c in drop_columns])
    ]

    missing_strategies = {}

    missing_custom_values = {}

    if missing_columns.empty:

        st.success("No missing values were found.")

    else:

        for index, row in missing_columns.iterrows():

            column = next(
                col for col in prepared_df.columns
                if str(col) == row["Column"]
            )

            with st.expander(
                f"{column} | {int(row['Missing']):,} missing "
                f"({row['Missing %']}%)"
            ):

                st.write(
                    f"**Data type:** {row['Data Type']}"
                )

                st.write(
                    f"**Suggested method:** "
                    f"{row['Missing Recommendation']}"
                )

                st.caption(
                    row["Recommendation Reason"]
                )

                options = [
                    "Keep missing",
                    "Mean",
                    "Median",
                    "Mode",
                    "Unknown",
                    "Custom value",
                    "Drop rows"
                ]

                selected = st.selectbox(
                    "Choose how to handle missing values",
                    options,
                    index=0,
                    key=f"missing_{column}"
                )

                missing_strategies[column] = selected

                column_is_numeric = pd.api.types.is_numeric_dtype(
                    prepared_df[column]
                )

                if selected in ["Mean", "Median"]:

                    if not column_is_numeric:

                        st.warning(
                            "This method requires a numeric column. "
                            "Select it for numeric conversion on the "
                            "Data Types & Text tab first."
                        )

                    else:

                        values = prepared_df[column].dropna()

                        if not values.empty:

                            st.write(
                                f"Mean: {values.mean():,.3f}"
                            )

                            st.write(
                                f"Median: {values.median():,.3f}"
                            )

                            fig = px.histogram(
                                values,
                                nbins=25,
                                title=f"Distribution of {column}"
                            )

                            fig.update_layout(height=280)

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                if selected == "Mode":

                    modes = prepared_df[column].mode(dropna=True)

                    if not modes.empty:

                        st.write(
                            f"Most frequent value: {modes.iloc[0]}"
                        )

                if selected == "Unknown":

                    if column_is_numeric:

                        st.warning(
                            "'Unknown' can only be used for text columns."
                        )

                    else:

                        st.caption(
                            "Missing values will be replaced with the "
                            "text 'Unknown'. Rows are kept."
                        )

                if selected == "Custom value":

                    custom_value = st.text_input(
                        "Value to use",
                        key=f"missing_custom_{column}"
                    )

                    missing_custom_values[column] = custom_value

                if selected == "Drop rows":

                    st.warning(
                        "Rows with missing values in this column "
                        "will be removed."
                    )

    add_imputed_flag = st.checkbox(
        "Add a column marking which values were filled "
        "(for example Quantity_was_missing)",
        value=False,
        key="add_imputed_flag"
    )

    st.caption(
        "Recommended when filled values feed into totals such as revenue: "
        "you can then see or exclude the estimated rows."
    )

    st.session_state["missing_strategies"] = (
        missing_strategies
    )

    st.session_state["missing_custom_values"] = (
        missing_custom_values
    )


# ==========================================================
# TAB 5: DUPLICATES AND OUTLIERS
# ==========================================================

with tab_duplicates:

    # Columns selected for dropping are not offered here.
    view_df = prepared_df.drop(
        columns=[
            col for col in st.session_state.get("drop_columns", [])
            if col in prepared_df.columns
        ]
    )

    # ------------------------------------------------------
    # DUPLICATE HANDLING
    # ------------------------------------------------------

    st.subheader("Duplicate detection")

    st.write(
        "Full-row duplicates and duplicate business identifiers "
        "are different. Select columns that uniquely identify a record. "
        "Fully identical rows are always removed."
    )

    full_duplicate_count = int(view_df.duplicated().sum())

    full_duplicate_pct = full_duplicate_count / len(view_df) * 100

    remove_full_duplicates = st.checkbox(
        f"Remove completely identical rows "
        f"({full_duplicate_count:,} found, {full_duplicate_pct:.1f}%)",
        value=True,
        key="remove_full_duplicates"
    )

    if full_duplicate_pct > 10:

        st.warning(
            f"{full_duplicate_pct:.0f}% of the rows are identical copies. "
            "If the dataset has no unique ID column (for example one row "
            "per booking), identical rows may be different real records "
            "that just look the same. Check before removing them."
        )

    duplicate_columns = st.multiselect(
        "Select identifier/key columns for duplicate checking",
        options=list(view_df.columns),
        key="duplicate_key_columns"
    )

    duplicate_action = st.selectbox(
        "Action for duplicate key records",
        [
            "Keep all",
            "Keep first",
            "Keep last",
            "Keep most complete"
        ],
        key="duplicate_action"
    )

    if duplicate_columns:

        duplicate_key_count = int(
            view_df.duplicated(
                subset=duplicate_columns,
                keep=False
            ).sum()
        )

        st.metric(
            "Rows belonging to duplicate key groups",
            duplicate_key_count
        )

        st.caption(
            "This is a preview. Text standardization can reveal "
            "additional duplicates during cleaning."
        )

        st.warning(
            "Duplicate key records may contain conflicting information. "
            "Review them before removing records."
        )

    else:

        st.caption(
            "No key-based duplicate check selected."
        )

    st.divider()

    # ------------------------------------------------------
    # OUTLIER DETECTION
    # ------------------------------------------------------

    st.subheader("Outlier detection")

    st.write(
        "The IQR method identifies potential extreme numerical values. "
        "It does not prove that those values are wrong."
    )

    outlier_candidates = [
        col
        for col in view_df.select_dtypes(include="number").columns
        if not is_identifier_column(col)
    ]

    outlier_report = detect_outliers_iqr(
        view_df,
        outlier_candidates
    )

    if outlier_report.empty:

        st.info(
            "No eligible numeric measurement columns were found."
        )

        default_outlier_columns = []

    else:

        st.dataframe(
            outlier_report.round(3),
            use_container_width=True,
            hide_index=True
        )

        default_outlier_columns = (
            outlier_report.loc[
                outlier_report["Outlier Count"] > 0,
                "Column"
            ].tolist()
        )

    outlier_columns = st.multiselect(
        "Select columns for outlier handling",
        outlier_candidates,
        default=default_outlier_columns,
        key="outlier_columns"
    )

    outlier_action = st.selectbox(
        "Choose outlier action",
        [
            "Keep outliers",
            "Cap outliers",
            "Remove outlier rows"
        ],
        key="outlier_action"
    )

    if outlier_action == "Keep outliers":

        st.caption(
            "Potential outliers will be reported but not modified."
        )

    elif outlier_action == "Cap outliers":

        st.warning(
            "Selected values outside the IQR boundaries will be capped."
        )

    else:

        st.warning(
            "Rows containing selected outliers will be removed. "
            "This may remove valid observations."
        )


# ==========================================================
# TAB 6: APPLY CLEANING AND RESULTS
# ==========================================================

with tab_apply:

    st.subheader("Review selected cleaning operations")

    st.write(
        "Review the settings below before applying them. "
        "Your original dataset will remain unchanged."
    )

    # ------------------------------------------------------
    # PREPARE SELECTED OPTIONS
    # ------------------------------------------------------

    missing_strategies = st.session_state.get(
        "missing_strategies", {}
    )

    missing_custom_values = st.session_state.get(
        "missing_custom_values", {}
    )

    numeric_columns = st.session_state.get(
        "numeric_columns", []
    )

    date_columns = st.session_state.get(
        "date_columns", []
    )

    identifier_text_columns = st.session_state.get(
        "identifier_text_columns", []
    )

    # Identifier conversions are handled separately from
    # ordinary numeric conversions.
    numeric_columns = [
        col for col in numeric_columns
        if col not in identifier_text_columns
    ]

    text_rules = st.session_state.get(
        "text_rules", {}
    )

    value_mappings = st.session_state.get(
        "value_mappings", {}
    )

    numeric_rules = st.session_state.get(
        "numeric_validation_rules", {}
    )

    date_rules = st.session_state.get(
        "date_relationship_rules", []
    )

    duplicate_columns = st.session_state.get(
        "duplicate_key_columns", []
    )

    duplicate_action = st.session_state.get(
        "duplicate_action", "Keep all"
    )

    outlier_columns = st.session_state.get(
        "outlier_columns", []
    )

    outlier_action = st.session_state.get(
        "outlier_action", "Keep outliers"
    )

    drop_columns = st.session_state.get(
        "drop_columns", []
    )

    remove_full_duplicates = st.session_state.get(
        "remove_full_duplicates", True
    )

    convert_placeholders = st.session_state.get(
        "convert_placeholders", True
    )

    date_dayfirst = (
        st.session_state.get("date_dayfirst_choice", DAY_FIRST_LABEL)
        == DAY_FIRST_LABEL
    )

    allow_conversion_loss = st.session_state.get(
        "allow_conversion_loss", False
    )

    add_imputed_flag = st.session_state.get(
        "add_imputed_flag", False
    )

    # ------------------------------------------------------
    # REVIEW SUMMARY
    # ------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Columns with missing values configured",
        len([
            strategy for strategy in missing_strategies.values()
            if strategy != "Keep missing"
        ])
    )

    c2.metric(
        "Numeric conversions",
        len(numeric_columns)
    )

    c3.metric(
        "Date conversions",
        len(date_columns)
    )

    st.write(
        f"**Columns to drop completely:** "
        f"{', '.join(map(str, drop_columns)) if drop_columns else 'None'}"
    )

    st.write(
        f"**Placeholders ('-', 'N/A') treated as missing:** "
        f"{'Yes' if convert_placeholders else 'No'}"
    )

    st.write(
        f"**Identifier columns converted to text:** "
        f"{len(identifier_text_columns)}"
    )

    st.write(
        f"**Text formatting rules:** {len(text_rules)}"
    )

    st.write(
        f"**Custom category mappings:** "
        f"{len([m for m in value_mappings.values() if m])} column(s)"
    )

    st.write(
        f"**Numeric validation rules:** {len(numeric_rules)}"
    )

    st.write(
        f"**Date relationship rules:** {len(date_rules)}"
    )

    st.write(
        f"**Duplicate key columns:** "
        f"{', '.join(map(str, duplicate_columns)) if duplicate_columns else 'None'}"
    )

    st.write(
        f"**Identical rows removed:** "
        f"{'Yes' if remove_full_duplicates else 'No'}"
    )

    st.write(
        f"**Duplicate action:** {duplicate_action}"
    )

    st.write(
        f"**Outlier action:** {outlier_action}"
    )

    # Show problems BEFORE the user applies anything.
    for line in prepare_log:

        if line.startswith("WARNING"):
            st.error(line)

    st.divider()

    # ------------------------------------------------------
    # APPLY BUTTON
    # ------------------------------------------------------

    if st.button(
        "Apply Cleaning Operations",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Cleaning your dataset..."):

            working_df = df.copy()

            # Convert identifiers to text without treating them
            # as numerical measurements.
            for column in identifier_text_columns:

                if column in working_df.columns:

                    working_df[column] = (
                        working_df[column]
                        .astype("string")
                    )

            # Save the input used for this round so the comparison
            # remains correct after the next Streamlit rerun.
            st.session_state["cleaning_round_source_df"] = working_df.copy()

            cleaned_df, cleaning_log = clean_dataset(
                df=working_df,
                missing_strategies=missing_strategies,
                numeric_columns=numeric_columns,
                date_columns=date_columns,
                text_rules=text_rules,
                value_mappings=value_mappings,
                numeric_validation_rules=numeric_rules,
                date_relationship_rules=date_rules,
                duplicate_columns=duplicate_columns,
                duplicate_action=duplicate_action,
                outlier_columns=outlier_columns,
                outlier_action=outlier_action,
                convert_placeholders=convert_placeholders,
                date_dayfirst=date_dayfirst,
                allow_conversion_loss=allow_conversion_loss,
                missing_custom_values=missing_custom_values,
                add_imputed_flag=add_imputed_flag,
                drop_columns=drop_columns,
                remove_full_duplicates=remove_full_duplicates
            )

            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["cleaning_log"] = cleaning_log
            st.session_state["cleaning_applied"] = True
            st.session_state["cleaning_round_number"] = (
                st.session_state.get("cleaning_round_number", 0) + 1
            )

            # Clear the previous round's selections on the next run,
            # so the user can configure a fresh cleaning round.
            st.session_state["_reset_cleaning_controls"] = True

        st.rerun()

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    cleaned_df = st.session_state.get("cleaned_df")

    cleaning_log = st.session_state.get("cleaning_log")

    if cleaned_df is not None:

        st.divider()

        st.subheader("Before and after comparison")

        st.caption(
            f"Cleaning round {st.session_state.get('cleaning_round_number', 1)}"
        )


        round_source_df = st.session_state.get(
            "cleaning_round_source_df"
        )

        if round_source_df is None:
            round_source_df = uploaded_df

        summary = final_quality_summary(
            round_source_df,
            cleaned_df
        )

        # --------------------------------------------------
        # SAFETY WARNINGS (data loss is never silent)
        # --------------------------------------------------

        for warning_text in quality_warnings(round_source_df, cleaned_df):
            st.warning(warning_text)

        before, after = st.columns(2)

        with before:

            st.markdown("**Input to this cleaning round**")

            st.metric(
                "Rows",
                f"{summary['Original Rows']:,}"
            )

            st.metric(
                "Missing values",
                f"{summary['Original Missing']:,}"
            )

            st.metric(
                "Full-row duplicates",
                f"{summary['Original Duplicates']:,}"
            )

        with after:

            st.markdown("**Result of this cleaning round**")

            st.metric(
                "Rows",
                f"{summary['Cleaned Rows']:,}",
                delta=summary["Cleaned Rows"] - summary["Original Rows"]
            )

            st.metric(
                "Missing values",
                f"{summary['Remaining Missing']:,}",
                delta=(
                    summary["Remaining Missing"]
                    - summary["Original Missing"]
                )
            )

            st.metric(
                "Full-row duplicates",
                f"{summary['Remaining Duplicates']:,}",
                delta=(
                    summary["Remaining Duplicates"]
                    - summary["Original Duplicates"]
                )
            )

        st.divider()

        # --------------------------------------------------
        # REMAINING MISSING VALUES
        # --------------------------------------------------

        st.subheader("Remaining missing values")

        remaining = (
            cleaned_df.isna()
            .sum()
            .reset_index()
        )

        remaining.columns = [
            "Column",
            "Remaining Missing"
        ]

        remaining = remaining[
            remaining["Remaining Missing"] > 0
        ]

        if remaining.empty:

            st.success(
                "No missing values remain."
            )

        else:

            st.warning(
                "Some missing values remain. They may have been "
                "intentionally preserved or created by validation "
                "and unsuccessful type conversion."
            )

            st.dataframe(
                remaining,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # --------------------------------------------------
        # CLEANING LOG
        # --------------------------------------------------

        st.subheader("Cleaning activity log")

        if cleaning_log:

            for item in cleaning_log:

                if item.startswith("WARNING"):
                    st.error(item)

                elif item.startswith("NOTE"):
                    st.info(item)

                else:
                    st.write(f"• {item}")

        st.divider()

        # --------------------------------------------------
        # FINAL DATA QUALITY CHECK
        # --------------------------------------------------

        st.subheader("Final data quality check")

        final_quality = analyze_quality(cleaned_df)

        st.dataframe(
            final_quality[
                [
                    "Column",
                    "Data Type",
                    "Missing",
                    "Hidden Missing",
                    "Missing %",
                    "Unique Values"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # --------------------------------------------------
        # CLEANED DATA PREVIEW
        # --------------------------------------------------

        st.subheader("Cleaned dataset preview")

        st.dataframe(
            cleaned_df.head(20),
            use_container_width=True
        )

        st.divider()

        # --------------------------------------------------
        # DOWNLOAD
        # --------------------------------------------------

        st.subheader("Export cleaned dataset")

        csv_data = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        excel_buffer = BytesIO()

        cleaned_df.to_excel(
            excel_buffer,
            index=False,
            engine="openpyxl"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name="datapilot_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:

            st.download_button(
                "Download Excel",
                data=excel_buffer.getvalue(),
                file_name="datapilot_cleaned_data.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True
            )

        st.caption(
            "The original uploaded dataset is preserved. New cleaning rounds use the latest cleaned result."
        )