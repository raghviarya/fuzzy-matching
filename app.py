# app.py

import io
from typing import List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tagging import classify_value

# Load .env (OPENAI_API_KEY etc.)
load_dotenv()

st.set_page_config(page_title="Fuzzy Mapper", layout="wide")

st.title("🔎 Fuzzy Mapping Tool")

st.markdown(
    """
Upload a dataset and a mapping list (e.g. divisions, categories, colours),
then use GPT-4o to tag each row in your chosen column.
"""
)

# --- Step 1: Upload files ---
st.subheader("1️⃣ Upload your files")

data_file = st.file_uploader(
    "Upload your main Excel file",
    type=["xlsx", "xls"],
    key="data_file",
)

mapping_source = st.radio(
    "How will you provide the mapping list?",
    ["Upload Excel", "Paste text list"],
    horizontal=True,
)

mapping_labels: List[str] = []

if mapping_source == "Upload Excel":
    mapping_file = st.file_uploader(
        "Upload your mapping Excel (one column with labels)",
        type=["xlsx", "xls"],
        key="mapping_file",
    )

    mapping_column_name = st.text_input(
        "Name of the column in the mapping file that contains the labels",
        value="label",
        help="The column that has your divisions/colours/etc. (e.g. 'division', 'label').",
    )

    if mapping_file is not None:
        try:
            mapping_df = pd.read_excel(mapping_file)
            if mapping_column_name not in mapping_df.columns:
                st.warning(
                    f"Column '{mapping_column_name}' not found in mapping file. "
                    f"Available columns: {list(mapping_df.columns)}"
                )
            else:
                mapping_labels = (
                    mapping_df[mapping_column_name]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )
        except Exception as e:
            st.error(f"Error reading mapping file: {e}")

else:
    mapping_text = st.text_area(
        "Paste your mapping list (one label per line)",
        placeholder="red\nblue\ngreen",
        height=150,
    )
    if mapping_text.strip():
        mapping_labels = [
            line.strip() for line in mapping_text.splitlines() if line.strip()
        ]

if mapping_labels:
    st.success(f"Loaded {len(mapping_labels)} labels: {mapping_labels}")

# --- Step 2: Parameters ---
st.subheader("2️⃣ Configure tagging")

col1, col2 = st.columns(2)

with col1:
    target_column = st.text_input(
        "Which column in the main Excel should be mapped?",
        placeholder="e.g. colours",
        help="This is the *only required* chatbot field.",
    )

with col2:
    allow_other = st.checkbox(
        'Allow "Other" as a tag',
        value=True,
        help="If checked, rows that don't fit any label can be tagged as 'Other'.",
    )

extra_context = st.text_area(
    "Additional context / instructions (optional)",
    placeholder=(
        "Explain what the column represents, any domain rules, or how you'd like "
        "ambiguous values handled."
    ),
    height=150,
)

model_choice = st.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o"],
    help="Use gpt-4o-mini for cheaper, faster tagging; gpt-4o for higher quality.",
)

new_col_name = st.text_input(
    "Name of the new mapped column",
    value="mapped_label",
)

# --- Step 3: Run tagging ---
st.subheader("3️⃣ Run mapping")

run_button = st.button("🚀 Start tagging", type="primary")

if run_button:
    if data_file is None:
        st.error("Please upload your main Excel file.")
    elif not mapping_labels:
        st.error("Please provide at least one mapping label (via file or text).")
    elif not target_column:
        st.error("Please specify which column in the Excel should be mapped.")
    else:
        try:
            df = pd.read_excel(data_file)
        except Exception as e:
            st.error(f"Error reading main Excel file: {e}")
            st.stop()

        if target_column not in df.columns:
            st.error(
                f"Column '{target_column}' not found in main file. "
                f"Available columns: {list(df.columns)}"
            )
            st.stop()

        st.info(
            "Tagging in progress... For large files, this may take a while in this MVP."
        )

        values = df[target_column].astype(str)

        mapped_tags = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(values)
        had_error = False

        for i, val in enumerate(values):
            try:
                tag = classify_value(
                    value=val,
                    mapping_labels=mapping_labels,
                    column_name=target_column,
                    allow_other=allow_other,
                    extra_context=extra_context,
                    model=model_choice,
                )
            except Exception as e:
                st.error(f"Error when tagging row {i+1}: {e}")
                had_error = True
                break

            mapped_tags.append(tag)
            if (i + 1) % 5 == 0 or i == total - 1:
                progress_bar.progress((i + 1) / total)
                status_text.text(f"Processed {i+1}/{total} rows...")

        if had_error:
            st.stop()

        # Attach mapped column
        df[new_col_name] = mapped_tags

        st.success("✅ Tagging complete!")

        st.subheader("Preview of tagged data")
        st.dataframe(df.head(20))

        # Create a downloadable Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="TaggedData")
        output.seek(0)

        st.download_button(
            label="📥 Download tagged Excel",
            data=output,
            file_name="tagged_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
