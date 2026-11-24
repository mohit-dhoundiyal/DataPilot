import streamlit as st
import pandas as pd
from engine.profiler import generate_profile

st.set_page_config(
    page_title="Data Profile",
    page_icon="🔍",
    layout='wide',
)

from utils.ui_styles import inject_css
inject_css()
st.title("Data Profiling")
st.write("Automatically understand your uploaded dataset")

df=st.session_state.get("df")

if df is None:
    st.warning("Please upload a dataset in the Home page first.")

else:
    profile = generate_profile(df)

    st.subheader("Dataset Overview")

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Total Rows",profile["total_rows"])
    col2.metric("Total Columns",profile["total_columns"])
    col3.metric("Missing Values",profile["total_missing_values"])
    col4.metric("Duplicate Rows",profile["duplicate_rows"])

    st.divider()

    st.subheader("Column Information")

    column_info = pd.DataFrame({
        "Column name":df.columns,
        "Data Type":profile["data_types"].values,
        "Missing Values":profile["missing_values"].values,
        "Unique Values":profile["unique_values"].values
    })

    st.dataframe(column_info,use_container_width=True)

    st.divider()

    st.subheader("Statistical Summary")

    st.dataframe(
        profile["numeric_summary"],
        use_container_width=True

    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)
