import streamlit as st
import json
import pathlib
import tempfile
import os
from dotenv import load_dotenv
from ocr_extractor import extract_page_json, merge_page_results, get_schema_descriptions, match_page_to_schema
from pdf2image import convert_from_path
from llm_handler import LLMHandler

st.set_page_config(page_title="Handwritten Form Extractor", page_icon="📝", layout="wide")
st.title("📝 Handwritten Form Extractor")
st.write("Upload a scanned PDF form and a JSON schema to extract handwritten content into structured JSON.")

# Load API key
load_dotenv()

try:
    llm = LLMHandler()
    model = llm.model
except Exception as e:
    st.error(f"⚠️ Failed to initialize LLM: {e}")
    st.stop()
    
uploaded_pdf = st.file_uploader("📄 Upload filled PDF form", type=["pdf"])
schema_dir = os.path.abspath("./schema")

if uploaded_pdf:
    temp_pdf_path = pathlib.Path(f"./temp_{uploaded_pdf.name}")

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    schema_files, schema_descriptions = get_schema_descriptions(schema_dir)

    with st.spinner("⚙️ Converting PDF pages..."):
        pages = convert_from_path(temp_pdf_path, dpi=150)
    st.success(f"✅ Converted {len(pages)} pages.")
    
    # Resize images to speed up processing and reduce memory usage
    max_width, max_height = 1024, 1024
    for i, img in enumerate(pages):
        img.thumbnail((max_width, max_height))

    st.subheader("🔍 Preview PDF Pages")
    include_pages = []
    cols = st.columns(4)
    for i, page in enumerate(pages, start=1):
        with cols[(i - 1) % 4]:
            st.image(page, caption=f"Page {i}")
            include = st.checkbox(f"Include Page {i}", value=True, key=f"include_{i}")
            include_pages.append(include)

    if st.button("✅ Confirm Pages and Extract"):
        selected_pages = [page for page, include in zip(pages, include_pages) if include]
        if not selected_pages:
            st.warning("No pages selected for extraction.")
        else:
            all_page_data = []
            matched_schemas = []
            progress = st.progress(0)
            status = st.empty()
            for i, page in enumerate(selected_pages, start=1):
                with st.spinner(f"⚙️ Matching selected page {i}/{len(selected_pages)} to schema ..."):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.close()
                        try:
                            page.save(tmp.name, "PNG")
                            with open(tmp.name, "rb") as img_file:
                                img_bytes = img_file.read()
                            schema_name = match_page_to_schema(llm, img_bytes, schema_descriptions)
                            if schema_name == "none":
                                st.warning(f"Page {i} does not match any schema. Skipping.")
                                continue
                            schema_path = pathlib.Path(schema_dir) / schema_name
                            if not schema_path.exists():
                                st.warning(f"Matched schema {schema_name} not found. Skipping page {i}.")
                                continue
                            matched_schemas.append(schema_path)
                            print(schema_path)
                            with open(schema_path, "r", encoding="utf-8") as f:
                                schema = json.load(f)
                            schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
                            try:
                                page_json = extract_page_json(llm, img_bytes, i, schema_text)
                            except Exception as e:
                                st.warning(f"⚠️ LLM error on page {i}: {e}")
                                page_json = {}
                            all_page_data.append(page_json)
                        finally:
                            try:
                                os.unlink(tmp.name)
                            except:
                                pass
                progress.progress(i / len(selected_pages))

            status.write("🧩 Merging all page results...")
            final_json = merge_page_results(all_page_data)
            st.success("✅ Extraction complete!")

            view_mode = st.radio("View extracted data as:", ["JSON Output", "Form UI View"])

            if view_mode == "JSON Output":
                st.subheader("📃 Extracted JSON")
                st.json(final_json)

                st.download_button(
                    label="⬇️ Download JSON",
                    file_name=f"{temp_pdf_path.stem}_extracted.json",
                    mime="application/json",
                    data=json.dumps(final_json, indent=2, ensure_ascii=False)
                )

            elif view_mode == "Form UI View":
                st.subheader("📋 Form View (Read-only)")
                st.caption("Extracted values displayed in form layout")

                for section, fields in final_json.items():
                    if isinstance(fields, dict):
                        st.markdown(f"### {section}")
                        for field, value in fields.items():
                            if isinstance(value, bool):
                                st.checkbox(field, value=value, disabled=True)
                            elif isinstance(value, list):
                                st.multiselect(field, options=value, default=value, disabled=True)
                            elif isinstance(value, dict):
                                st.markdown(f"**{field}:**")
                                for subfield, subval in value.items():
                                    st.text_input(f"{field} → {subfield}", value=subval or "", disabled=True)
                            else:
                                st.text_input(field, value=value or "", disabled=True)
                    else:
                        st.text_input(section, value=fields or "", disabled=True)
