import streamlit as st
import json
import pathlib
import tempfile
import os
from dotenv import load_dotenv
from ocr_extractor import extract_page_json, merge_page_results, get_all_schema, match_page_to_schema
from pdf2image import convert_from_path
from llm_handler import LLMHandler
import io
import pandas as pd
from pandas import ExcelWriter
import time

def echo_llm_test(llm):
    st.header("🔊 Echo LLM Test (Debug)")
    st.write("Upload an image and enter a prompt to send directly to the LLM. Useful for debugging model/image input.")

    with st.expander("LLM Options (Advanced)", expanded=False):
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.01, key="echo_temperature")
        top_p = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="echo_top_p")
        num_ctx = st.number_input("Context Length (num_ctx)", min_value=512, max_value=32768, value=16000, step=1, key="echo_num_ctx")
        num_predict = st.number_input("Max Tokens (num_predict)", min_value=1, max_value=8192, value=6000, step=1, key="echo_num_predict")

    uploaded_img = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="echo_img")
    prompt_text = st.text_area("Prompt to send to LLM", value="Describe the image and then repeat this sentence: Hello world.", key="echo_prompt")
    if uploaded_img and st.button("Run Echo Test", key="run_echo"):
        img_bytes = uploaded_img.read()
        st.info("Sending image and prompt to LLM...")
        try:
            import base64
            image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            response = llm.model.generate(
                model=llm.model_name,
                prompt=prompt_text,
                images=[image_b64],
                stream=False,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_ctx": int(num_ctx),
                    "num_predict": int(num_predict),
                }
            )
            st.subheader("LLM Raw Response")
            st.code(response, language="json")
            st.subheader("LLM Response")
            st.code(response.get('response', ''), language="json")
        except Exception as e:
            st.error(f"Echo test error: {e}")

st.set_page_config(page_title="Handwritten Form Extractor", page_icon="📝", layout="wide")
st.title("📝 Handwritten Form Extractor")
st.write("Upload a scanned PDF form and a JSON schema to extract handwritten content into structured JSON.")

MODEL_OPTIONS = [
    "llama3.2-vision:11b",
    "minicpm-v:8b",
    "qwen3-vl:2b",
    "qwen2.5vl:3b",
    "qwen3-vl:4b-instruct-q8_0"
]

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

model_disabled = st.session_state.uploaded
selected_model = st.selectbox("Select LLM Model", MODEL_OPTIONS, index=0, disabled=model_disabled)

analytics_disabled = st.session_state.uploaded
analytics_enabled = st.checkbox(
    "Enable Analytics", 
    value=st.session_state.get("analytics_enabled", True), 
    disabled=analytics_disabled,
    key="analytics_checkbox"
)
st.session_state.analytics_enabled = analytics_enabled
os.environ["ANALYTICS_ENABLED"] = str(analytics_enabled)

load_dotenv()
DEBUG = os.getenv("DEBUG", "False") == "True"
ANALYTICS_ENABLED = os.environ.get("ANALYTICS_ENABLED", "False") == "True"

try:
    os.environ["LLM_MODEL_NAME"] = selected_model
    llm = LLMHandler()
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

with st.expander("🔊 Echo LLM Test (Debug)", expanded=False):
    echo_llm_test(llm)

if not st.session_state.uploaded:
    uploaded_pdf = st.file_uploader("📄 Upload filled PDF form", type=["pdf"])
    if uploaded_pdf:
        st.session_state.uploaded = True
        st.session_state.uploaded_pdf_name = uploaded_pdf.name
        temp_pdf_path = pathlib.Path(f"./temp_{uploaded_pdf.name}")
        try:
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())
        except Exception as e:
            st.error(f"Error saving uploaded PDF: {e}")
            st.stop()
        st.rerun()
else:
    temp_pdf_path = pathlib.Path(f"./temp_{st.session_state.uploaded_pdf_name}")
    schema_dir = os.path.abspath("./schema")
    try:
        schema_files = get_all_schema(schema_dir)
    except Exception as e:
        st.error(f"Error loading schemas: {e}")
        st.stop()

    try:
        with st.spinner("⚙️ Converting PDF pages..."):
            pages = convert_from_path(temp_pdf_path, dpi=150)
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        st.stop()

    if "extraction_started" not in st.session_state or not st.session_state.extraction_started:
        st.success(f"✅ Converted {len(pages)} pages.")

    max_width, max_height = 1024, 1024
    for img in pages:
        img.thumbnail((max_width, max_height))

    if "extraction_started" not in st.session_state:
        st.session_state.extraction_started = False

    if not st.session_state.extraction_started:
        st.subheader("🔍 Preview PDF Pages")
        preview_container = st.container()
        include_pages = []
        cols = preview_container.columns(5)
        for i, page in enumerate(pages, start=1):
            with cols[(i - 1) % 5]:
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

            if "schema_matches" not in st.session_state:
                st.session_state.schema_matches = {}
            schema_matches = st.session_state.schema_matches

            matched_schema_names = []
            matched_schema_paths = []
            matched_img_bytes = []
            prev_schema_num = None
            for i, page in enumerate(selected_pages, start=1):
                start_time = time.time()
                with st.spinner(f"⚙️ Matching selected page {i}/{len(selected_pages)} to schema ..."):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            page.save(tmp.name, "PNG")
                            with open(tmp.name, "rb") as img_file:
                                img_bytes = img_file.read()
                        cache_key = f"page_{i}_schema"
                        if cache_key in schema_matches:
                            schema_name = schema_matches[cache_key]
                        else:
                            try:
                                schema_name = match_page_to_schema(
                                    llm, img_bytes, schema_files, page_index=i, prev_schema_num=prev_schema_num
                                )
                                schema_matches[cache_key] = schema_name
                                st.session_state.schema_matches = schema_matches
                            except Exception as e:
                                st.error(f"⚠️ Error matching schema: {e}")
                                st.stop()
                        if schema_name == "none" or not schema_name:
                            st.warning(f"Page {i} does not match any schema. Skipping.")
                            matched_schema_names.append(None)
                            matched_schema_paths.append(None)
                            matched_img_bytes.append(None)
                            prev_schema_num = None
                        else:
                            schema_path = pathlib.Path(schema_dir) / schema_name
                            if not schema_path.exists():
                                st.warning(f"Matched schema {schema_name} not found. Skipping page {i}.")
                                matched_schema_names.append(None)
                                matched_schema_paths.append(None)
                                matched_img_bytes.append(None)
                                prev_schema_num = None
                            else:
                                matched_schema_names.append(schema_name)
                                matched_schema_paths.append(schema_path)
                                matched_img_bytes.append(img_bytes)
                                try:
                                    prev_schema_num = int(''.join(filter(str.isdigit, schema_name)))
                                except Exception:
                                    prev_schema_num = None
                    except Exception as e:
                        st.error(f"⚠️ Error processing page {i}: {e}")
                        matched_schema_names.append(None)
                        matched_schema_paths.append(None)
                        matched_img_bytes.append(None)
                        prev_schema_num = None
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except Exception:
                            pass
                end_time = time.time()
                progress.progress(i / len(selected_pages))
            status.write("✅ Schema matching complete.")

            if DEBUG:
                st.subheader("🔎 Debug: Page-to-Schema Assignment")
                debug_rows = []
                for i, schema_name in enumerate(matched_schema_names, start=1):
                    debug_rows.append({
                        "Page": i,
                        "Schema File": schema_name if schema_name else "(none)"
                    })
                st.dataframe(debug_rows)

            for i, (schema_name, schema_path, img_bytes) in enumerate(zip(matched_schema_names, matched_schema_paths, matched_img_bytes), start=1):
                if not schema_name or not schema_path or not img_bytes:
                    continue
                if "page_extracted_data" not in st.session_state:
                    st.session_state.page_extracted_data = {}
                page_cache_key = f"page_{i}_extracted"
                if page_cache_key in st.session_state.page_extracted_data:
                    page_json = st.session_state.page_extracted_data[page_cache_key]
                else:
                    start_time = time.time()
                    with st.spinner(f"⚙️ Extracting data from page {i}/{len(selected_pages)} ..."):
                        try:
                            matched_schemas.append(schema_path)
                            try:
                                with open(schema_path, "r", encoding="utf-8") as f:
                                    schema = json.load(f)
                            except Exception as e:
                                st.warning(f"⚠️ Error loading schema file {schema_name}: {e}")
                                continue
                            schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
                            try:
                                page_json = extract_page_json(llm, img_bytes, i, schema_text)
                            except Exception as e:
                                st.warning(f"⚠️ LLM error on page {i}: {e}")
                                page_json = {}
                            st.session_state.page_extracted_data[page_cache_key] = page_json
                        except Exception as e:
                            st.error(f"⚠️ Error extracting data from page {i}: {e}")
                            continue
                    end_time = time.time()
                    elapsed = end_time - start_time
                    page_times.append(elapsed)
                    page_stats.append({
                        "page": i,
                        "time_sec": elapsed,
                        "schema": schema_name,
                        "fields_extracted": len(page_json) if isinstance(page_json, dict) else 0
                    })
                    progress.progress(i / len(selected_pages))
                    if DEBUG:
                        avg_time = sum(page_times) / len(page_times)
                        st.info(
                            f"Page {i} extracted in {elapsed:.2f}s | "
                            f"Avg: {avg_time:.2f}s | "
                            f"Fields: {page_stats[-1]['fields_extracted']} | "
                            f"Schema: {page_stats[-1]['schema']}"
                        )
                if page_cache_key in st.session_state.page_extracted_data:
                    page_json = st.session_state.page_extracted_data[page_cache_key]
                    if len(page_times) < i:
                        page_times.append(0)
                        page_stats.append({
                            "page": i,
                            "time_sec": 0,
                            "schema": schema_name,
                            "fields_extracted": len(page_json) if isinstance(page_json, dict) else 0
                        })
                    progress.progress(i / len(selected_pages))
                all_page_data.append(page_json)

            status.write("🧩 Merging all page results...")
            try:
                final_json = merge_page_results(all_page_data)
            except Exception as e:
                st.error(f"Error merging page results: {e}")
                st.stop()
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

            def get_json_edit_key(path):
                return "json_edit_" + "_".join(str(p) for p in path)

            def render_json_stack(data, path=[]):
                if isinstance(data, dict):
                    for k, v in data.items():
                        with st.expander(str(k), expanded=False):
                            render_json_stack(v, path + [k])
                elif isinstance(data, list):
                    for idx, item in enumerate(data):
                        with st.expander(f"Item {idx+1}", expanded=False):
                            render_json_stack(item, path + [str(idx)])
                else:
                    edit_key = get_json_edit_key(path)
                    default_value = data if data is not None else ""
                    if isinstance(default_value, bool):
                        st.checkbox("Value", value=st.session_state.get(edit_key, default_value), key=edit_key)
                    elif isinstance(default_value, str) and default_value in ("True", "False"):
                        bool_val = default_value == "True"
                        st.checkbox("Value", value=st.session_state.get(edit_key, bool_val), key=edit_key)
                    else:
                        st.text_input("Value", value=st.session_state.get(edit_key, default_value), key=edit_key)

            def update_json_with_edits(data, path=[]):
                if isinstance(data, dict):
                    return {k: update_json_with_edits(v, path + [k]) for k, v in data.items()}
                elif isinstance(data, list):
                    return [update_json_with_edits(item, path + [i]) for i, item in enumerate(data)]
                else:
                    edit_key = get_json_edit_key(path)
                    val = st.session_state.get(edit_key, data)
                    if isinstance(data, bool):
                        return bool(val)
                    elif isinstance(data, str) and data in ("True", "False"):
                        return "True" if val else "False"
                    else:
                        return val

            st.subheader("🗂️ Extracted Data Explorer")
            st.caption("Click to expand sections. Properties are stacked vertically. Deepest fields are editable.")
            try:
                render_json_stack(final_json)
            except Exception as e:
                st.error(f"Error rendering JSON explorer: {e}")

            edited_json = update_json_with_edits(final_json)

            def json_to_excel(json_data):
                try:
                    df = pd.json_normalize(json_data, sep=".")
                    output = io.BytesIO()
                    with ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name="Form-Extraction")
                    return output.getvalue()
                except Exception as e:
                    st.error(f"Error exporting to Excel: {e}")
                    return b""

            col_json, col_excel = st.columns(2)
            with col_json:
                try:
                    st.download_button(
                        label="⬇️ Download JSON",
                        file_name=f"{temp_pdf_path.stem}_extracted.json",
                        mime="application/json",
                        data=json.dumps(edited_json, indent=2, ensure_ascii=False)
                    )
                except Exception as e:
                    st.error(f"Error preparing JSON download: {e}")
            with col_excel:
                try:
                    excel_bytes = json_to_excel(edited_json)
                    st.download_button(
                        label="⬇️ Download Excel",
                        file_name=f"{temp_pdf_path.stem}_extracted.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        data=excel_bytes
                    )
                except Exception as e:
                    st.error(f"Error preparing Excel download: {e}")

            if st.button("Reset"):
                st.session_state.clear()
                st.rerun()
