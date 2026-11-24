
import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(
    page_title="DataPilot",
    page_icon="📊",
    layout="wide"
)

from utils.ui_styles import inject_css
inject_css()

st.title("📊 DataPilot")
st.subheader("AI-Assisted Data Analytics Platform")

st.write(
    "Upload your dataset to profile, clean, analyze "
    "and visualize your data."
)

if "df" not in st.session_state:
    st.session_state["df"] = None

if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"],
    key="dataset_uploader"
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))

        st.session_state["df"] = df.copy()
        st.session_state["uploaded_filename"] = uploaded_file.name

    except Exception as e:
        st.error(f"Unable to read the dataset: {e}")

df = st.session_state["df"]

if isinstance(df, pd.DataFrame):
    st.success(
        f"Dataset available: {st.session_state['uploaded_filename']}"
        if st.session_state["uploaded_filename"]
        else "Dataset loaded successfully!"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(column_info, use_container_width=True)
