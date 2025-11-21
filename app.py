import streamlit as st
import json
import pathlib
import tempfile
import os
from dotenv import load_dotenv
from ocr_extractor import extract_page_json, merge_page_results, get_schema_descriptions, match_page_to_schema
from pdf2image import convert_from_path
from llm_handler import LLMHandler
import io
import pandas as pd
from pandas import ExcelWriter
import time

try:
    st.set_page_config(page_title="Handwritten Form Extractor", page_icon="📝", layout="wide")
    st.title("📝 Handwritten Form Extractor")
    st.write("Upload a scanned PDF form and a JSON schema to extract handwritten content into structured JSON.")

    MODEL_OPTIONS = [
        "minicpm-v:8b",
        "llama3.2-vision:11b",
        "qwen3-vl:2b",
        "qwen2.5-vl:3b"
    ]
    if "uploaded" not in st.session_state:
        st.session_state.uploaded = False

    model_disabled = st.session_state.uploaded
    selected_model = st.selectbox("Select LLM Model", MODEL_OPTIONS, index=0, disabled=model_disabled)

    analytics_disabled = st.session_state.uploaded
    analytics_enabled = st.checkbox(
        "Enable Analytics", 
        value=st.session_state.get("analytics_enabled", False), 
        disabled=analytics_disabled,
        key="analytics_checkbox"
    )
    st.session_state.analytics_enabled = analytics_enabled
    os.environ["ANALYTICS_ENABLED"] = str(analytics_enabled)

    load_dotenv()
    DEBUG = os.getenv("DEBUG", "0") == "1"
    ANALYTICS_ENABLED = os.environ.get("ANALYTICS_ENABLED", "False") == "True"

    try:
        os.environ["LLM_MODEL_NAME"] = selected_model
        llm = LLMHandler()
        model = llm.model
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()

    if not st.session_state.uploaded:
        uploaded_pdf = st.file_uploader("📄 Upload filled PDF form", type=["pdf"])
        if uploaded_pdf:
            st.session_state.uploaded = True
            st.session_state.uploaded_pdf_name = uploaded_pdf.name
            temp_pdf_path = pathlib.Path(f"./temp_{uploaded_pdf.name}")
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())
            st.rerun()
    else:
        temp_pdf_path = pathlib.Path(f"./temp_{st.session_state.uploaded_pdf_name}")
        schema_dir = os.path.abspath("./schema")
        schema_files, schema_descriptions = get_schema_descriptions(schema_dir)

        with st.spinner("⚙️ Converting PDF pages..."):
            pages = convert_from_path(temp_pdf_path, dpi=150)
        if "extraction_started" not in st.session_state or not st.session_state.extraction_started:
            st.success(f"✅ Converted {len(pages)} pages.")

        max_width, max_height = 1024, 1024
        for i, img in enumerate(pages):
            img.thumbnail((max_width, max_height))

        if "extraction_started" not in st.session_state:
            st.session_state.extraction_started = False

        if not st.session_state.extraction_started:
            st.subheader("🔍 Preview PDF Pages")
            preview_container = st.container()
            include_pages = []
            cols = preview_container.columns(4)
            for i, page in enumerate(pages, start=1):
                with cols[(i - 1) % 4]:
                    st.image(page, caption=f"Page {i}")
                    include = st.checkbox(f"Include Page {i}", value=True, key=f"include_{i}")
                    include_pages.append(include)

            if st.button("✅ Confirm Pages and Extract"):
                st.session_state.extraction_started = True
                st.session_state.include_pages = include_pages
                st.rerun()
            if st.button("Cancel"):
                st.session_state.clear()
                st.rerun()
        else:
            include_pages = st.session_state.include_pages
            selected_pages = [page for page, include in zip(pages, include_pages) if include]
            if not selected_pages:
                st.warning("No pages selected for extraction.")
            else:
                all_page_data = []
                matched_schemas = []
                progress = st.progress(0)
                status = st.empty()
                page_times = []
                page_stats = []
                for i, page in enumerate(selected_pages, start=1):
                    start_time = time.time()
                    with st.spinner(f"⚙️ Matching selected page {i}/{len(selected_pages)} to schema ..."):
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            tmp.close()
                            try:
                                page.save(tmp.name, "PNG")
                                with open(tmp.name, "rb") as img_file:
                                    img_bytes = img_file.read()
                                try:
                                    schema_name = match_page_to_schema(llm, img_bytes, schema_descriptions)
                                except Exception as e:
                                    st.error(f"⚠️ {e}")
                                    st.stop()
                                if schema_name == "none":
                                    st.warning(f"Page {i} does not match any schema. Skipping.")
                                    continue
                                schema_path = pathlib.Path(schema_dir) / schema_name
                                if not schema_path.exists():
                                    st.warning(f"Matched schema {schema_name} not found. Skipping page {i}.")
                                    continue
                                matched_schemas.append(schema_path)
                                with open(schema_path, "r", encoding="utf-8") as f:
                                    schema = json.load(f)
                                schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
                                try:
                                    page_json = extract_page_json(llm, img_bytes, i, schema_text)
                                except Exception as e:
                                    st.warning(f"⚠️ LLM error on page {i}: {e}")
                                    page_json = {}
                                all_page_data.append(page_json)
                            except Exception as e:
                                st.error(f"⚠️ {e}")
                                st.stop()
                            finally:
                                try:
                                    os.unlink(tmp.name)
                                except:
                                    pass
                    end_time = time.time()
                    elapsed = end_time - start_time
                    page_times.append(elapsed)
                    page_stats.append({
                        "page": i,
                        "time_sec": elapsed,
                        "schema": schema_name if 'schema_name' in locals() else "none",
                        "fields_extracted": len(page_json) if isinstance(page_json, dict) else 0
                    })
                    progress.progress(i / len(selected_pages))
                    if DEBUG:
                        avg_time = sum(page_times) / len(page_times)
                        st.info(
                            f"Page {i} processed in {elapsed:.2f}s | "
                            f"Avg: {avg_time:.2f}s | "
                            f"Fields: {page_stats[-1]['fields_extracted']} | "
                            f"Schema: {page_stats[-1]['schema']}"
                        )

                status.write("🧩 Merging all page results...")
                final_json = merge_page_results(all_page_data)
                st.success("✅ Extraction complete!")

                if ANALYTICS_ENABLED:
                    st.subheader("📊 Extraction Statistics Summary")
                    total_time = sum(page_times)
                    avg_time = total_time / len(page_times) if page_times else 0
                    st.markdown(
                        f"- **Total pages processed:** {len(page_times)}\n"
                        f"- **Total extraction time:** {total_time:.2f}s\n"
                        f"- **Average page time:** {avg_time:.2f}s\n"
                        f"- **Fields extracted per page:** {[stat['fields_extracted'] for stat in page_stats]}"
                    )
                    st.dataframe(pd.DataFrame(page_stats))

                def render_json_tabs(data, path=[]):
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        if keys:
                            tab_labels = [str(k) for k in keys]
                            tabs = st.tabs(tab_labels)
                            for i, key in enumerate(keys):
                                with tabs[i]:
                                    render_json_tabs(data[key], path + [key])
                        else:
                            st.write("No data.")
                    elif isinstance(data, list):
                        for idx, item in enumerate(data):
                            st.markdown(f"**Item {idx+1}:**")
                            render_json_tabs(item, path + [str(idx)])
                    else:
                        st.markdown(f"**Value:** {data if data is not None else ''}")

                st.subheader("🗂️ Extracted Data Explorer")
                st.caption("Click tabs to drill down into nested sections. Deepest fields show their values.")
                render_json_tabs(final_json)

                def json_to_excel(json_data):
                    # TODO Placeholder: implement mapping values then exporting to Excel
                    df = pd.json_normalize(json_data, sep=".")
                    output = io.BytesIO()
                    with ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name="Form-Extraction")
                    return output.getvalue()

                # Download buttons
                col_json, col_excel = st.columns(2)
                with col_json:
                    st.download_button(
                        label="⬇️ Download JSON",
                        file_name=f"{temp_pdf_path.stem}_extracted.json",
                        mime="application/json",
                        data=json.dumps(final_json, indent=2, ensure_ascii=False)
                    )
                with col_excel:
                    excel_bytes = json_to_excel(final_json)
                    st.download_button(
                        label="⬇️ Download Excel",
                        file_name=f"{temp_pdf_path.stem}_extracted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        data=excel_bytes
                    )

                if st.button("Reset"):
                    for key in st.session_state.keys():
                        del st.session_state[key]
                    st.experimental_rerun()
except Exception as e:
    st.error(f"⚠️ Unhandled error: {e}")
