from io import BytesIO
import streamlit as st
import json
import time
import csv
from pathlib import Path
from pdf2image import convert_from_path
from llm_handler import LLMHandler
from ocr_extractor import extract_from_pdf, count_filled_fields, get_all_schema, match_page_to_schema

from field_transformer import (
    apply_mapping,
    get_property_names_in_order,
    load_mapping_config,
    preview_mapping
)

st.set_page_config(page_title="TOT Form Extractor", page_icon="📄", layout="wide")

def default_session_state():
    st.session_state.extraction_results = None
    st.session_state.edited_data = None
    st.session_state.view_mode = "JSON View"
    st.session_state.data_appended = False
    st.session_state.pdf_uploaded = False
    st.session_state.pdf_uploaded_name = None
    st.session_state.pdf_pages = None
    st.session_state.include_pages = []
    st.session_state.page_selection_confirmed = False
    st.session_state.extraction_complete = False
    st.session_state.schema_matches_confirmed = False
    st.session_state.matched_schemas = None

# Initialize session state
if 'initialized' not in st.session_state:
    default_session_state()
    st.session_state.split_schema = False
    st.session_state.initialized = True


st.title("📄 TOT Form Data Extractor")
st.markdown("Extract structured data from PDF forms using Ollama Vision Models")

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_pdf = st.file_uploader("Upload PDF Form", type=['pdf'], key="pdf_uploader")
    
    # If the file uploader is cleared (file removed with 'x'), reset session state
    if not uploaded_pdf and st.session_state.pdf_uploaded:
        default_session_state()
    
    # Schema mode selector
    use_split_schema = st.checkbox(
        "Use Split Schema (per-page matching)",
        key="split_schema",
        help="Enable to match each page to different schema files (schema1.json, schema2.json, etc.)"
    )
    
    if st.session_state.split_schema:
        # For split schema mode, use schema directory
        schema_dir_path = Path(__file__).parent.parent / "schema"
        if schema_dir_path.exists():
            schema_files_count = len(list(schema_dir_path.glob("schema*.json")))
            st.info(f"📁 Using schema directory with {schema_files_count} schema files")
        else:
            st.warning(f"⚠️ Schema directory not found: {schema_dir_path}")
        schema_file = None
    else:
        # For single schema mode, allow upload
        schema_file = st.file_uploader("Upload Schema (optional)", type=['json'])
    
    # Model selector dropdown
    model_name = st.selectbox(
        "Select Ollama Model",
        options=["qwen2.5vl:3b","minicpm-v:8b", "gemma3:4b", "llama3.2-vision:11b", "qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:4b-instruct-q8_0", "bakllava:7b"],
        index=0,
        help="Choose the vision model to use for extraction"
    )
    
    st.info(f"🤖 Using: **{model_name}**")
    
# Handle PDF upload and conversion (OUTSIDE tabs to avoid re-running)
if uploaded_pdf:
    uploadChanged = st.session_state.pdf_uploaded_name != uploaded_pdf.name
    if (not st.session_state.pdf_uploaded or uploadChanged):
        # If a different PDF is uploaded, reset state
        if uploadChanged:
            default_session_state()
        # Save uploaded file temporarily
        pdf_path = Path("temp_upload.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        
        # Convert to images for preview
        with st.spinner("⚙️ Converting PDF pages..."):
            try:
                pages = convert_from_path(pdf_path, dpi=150)
                # Thumbnail the images
                max_width, max_height = 1024, 1024
                for img in pages:
                    img.thumbnail((max_width, max_height))
                
                st.session_state.pdf_pages = pages
                st.session_state.pdf_uploaded = True
                st.session_state.pdf_uploaded_name = uploaded_pdf.name
                st.session_state.page_selection_confirmed = False
                st.success(f"✅ Converted {len(pages)} pages.")
            except Exception as e:
                st.error(f"Error converting PDF to images: {e}")
                st.stop()

# Main area tabs
tab1, tab2, tab3 = st.tabs(["📊 Extraction", "✏️ Edit & Approve", "💾 Export"])

with tab1:
    st.header("Extraction Results")
    
    # Show message if no PDF uploaded yet
    if not st.session_state.pdf_uploaded:
        st.info("👆 Upload and extract a PDF first to view extraction results.")
    
    # Show page preview and selection if PDF is uploaded but not confirmed
    if st.session_state.pdf_uploaded and not st.session_state.page_selection_confirmed:
        st.subheader("🔍 Preview PDF Pages")
        st.info("Select which pages to include in the extraction")
        
        pages = st.session_state.pdf_pages
        if len(st.session_state.include_pages) != len(pages):
            st.session_state.include_pages = [True] * len(pages)
        
        # Display pages in a grid
        cols_per_row = 4
        for i in range(0, len(pages), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                page_idx = i + j
                if page_idx < len(pages):
                    with cols[j]:
                        st.image(pages[page_idx], caption=f"Page {page_idx + 1}")
                        include = st.checkbox(
                            f"Include Page {page_idx + 1}", 
                            value=st.session_state.include_pages[page_idx], 
                            key=f"include_{page_idx + 1}"
                        )
                        st.session_state.include_pages[page_idx] = include
        
        # Confirmation buttons
        if st.button("✅ Confirm Pages and Extract", width="stretch"):
            st.session_state.page_selection_confirmed = True
            st.session_state.selected_pages = st.session_state.include_pages.copy()
            st.rerun()
    
    # Schema matching and confirmation
    if (
        st.session_state.page_selection_confirmed
        and st.session_state.pdf_pages
        and st.session_state.split_schema
        and not st.session_state.schema_matches_confirmed
    ):
        pdf_path = Path("temp_upload.pdf")
        selected_page_indices = [i for i, include in enumerate(st.session_state.selected_pages) if include]
        pages = st.session_state.pdf_pages
        selected_pages = [pages[i] for i in selected_page_indices]
        schema_dir = Path(__file__).parent.parent / "schema"
        schema_files = list(get_all_schema(str(schema_dir)))
        schema_file_names = [f.name for f in schema_files]

        # Run schema matching if not already done
        if st.session_state.matched_schemas is None:
            st.info("🔍 Matching each selected page to a schema. Please wait...")
            matched_schemas = []
            for idx, img in enumerate(selected_pages, start=1):
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                schema_name = match_page_to_schema(
                    LLMHandler(model_name=model_name), img_bytes, schema_files
                )
                # fallback to first schema if no match
                if schema_name == "none" or schema_name == "" or not schema_name:
                    schema_name = schema_file_names[0] if schema_file_names else ""
                matched_schemas.append(schema_name)
            st.session_state.matched_schemas = matched_schemas
            st.rerun()

        # UI for user to confirm/override schema matches
        st.subheader("🔗 Confirm Schema Matches for Each Page")
        st.info("Review the matched schema for each page. You can override the selection if needed, then confirm to proceed with extraction.")

        # Prepare editable schema assignments
        schema_assignments = []
        for idx, (img, default_schema) in enumerate(zip(selected_pages, st.session_state.matched_schemas)):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(img, caption=f"Page {selected_page_indices[idx]+1}", width=180)
            with col2:
                schema_choice = st.selectbox(
                    f"Schema for Page {selected_page_indices[idx]+1}",
                    schema_file_names,
                    index=schema_file_names.index(default_schema) if default_schema in schema_file_names else 0,
                    key=f"schema_select_{idx}"
                )
                schema_assignments.append(schema_choice)

        # Confirm/cancel buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Schema Assignments", key="confirm_schema_assignments"):
                st.session_state.matched_schemas = schema_assignments
                st.session_state.schema_matches_confirmed = True
                st.rerun()
        with col2:
            if st.button("❌ Back to Page Selection", key="back_to_page_selection"):
                st.session_state.page_selection_confirmed = False
                st.session_state.matched_schemas = None
                st.session_state.schema_matches_confirmed = False
                st.rerun()
        st.stop()

    # --- Extraction after schema confirmation ---
    if (
        st.session_state.page_selection_confirmed
        and st.session_state.pdf_pages
        and (
            (not st.session_state.split_schema and not st.session_state.extraction_complete)
            or (st.session_state.split_schema and st.session_state.schema_matches_confirmed and not st.session_state.extraction_complete)
        )
    ):
        pdf_path = Path("temp_upload.pdf")
        selected_page_indices = [i for i, include in enumerate(st.session_state.selected_pages) if include]
        
        # Get selected pages
        selected_page_indices = [i for i, include in enumerate(st.session_state.selected_pages) if include]
        
        if not selected_page_indices:
            st.warning("No pages selected for extraction.")
        else:
            st.info(f"Processing {len(selected_page_indices)} selected pages...")
            
            # Run extraction with progress
            with st.spinner("🔄 Processing PDF..."):
                start_time = time.time()
                
                try:
                    if st.session_state.split_schema:
                        schema_dir = Path(__file__).parent.parent / "schema"
                        if not schema_dir.exists():
                            st.error(f"❌ Schema directory not found: {schema_dir}")
                            st.stop()
                        # Use user-confirmed schema assignments
                        matched_schema_names = st.session_state.matched_schemas
                        results, timings = extract_from_pdf(
                            str(pdf_path),
                            schema_path=None,
                            output_path=None,
                            model_name=model_name,
                            use_split_schema=True,
                            schema_dir=str(schema_dir),
                            selected_page_indices=selected_page_indices,
                            matched_schema_names=matched_schema_names
                        )
                    else:
                        # Single schema mode - use one schema for all pages
                        schema_path = "ocr_schema.json"
                        if schema_file:
                            schema_path = "temp_schema.json"
                            with open(schema_path, "w") as f:
                                f.write(schema_file.read().decode())
                        
                        results, timings = extract_from_pdf(
                            str(pdf_path), 
                            schema_path, 
                            output_path=None,
                            model_name=model_name,
                            use_split_schema=False,
                            selected_page_indices=selected_page_indices
                        )
                    
                    st.session_state.extraction_results = results
                    st.session_state.edited_data = None  # Reset edited data
                    st.session_state.data_appended = False  # Reset append flag for new document
                    st.session_state.extraction_complete = True  # Mark extraction as complete
                    
                    # Display model used and schema mode
                    schema_mode = results['metadata'].get('schema_mode', 'single')
                    st.info(f"🤖 Extracted using model: **{results['metadata'].get('model_name', 'Unknown')}** | Schema mode: **{schema_mode}**")
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("⏱️ Total Time", f"{results['metadata']['total_time_seconds']}s")
                    
                    with col2:
                        st.metric("📄 Pages", results['metadata']['total_pages'])
                    
                    with col3:
                        st.metric("⚡ Avg Time/Page", f"{results['metadata']['average_time_per_page']}s")
                    
                    with col4:
                        st.metric("📝 Filled Fields", results['metadata']['total_filled_fields'])
                    
                    st.success("✅ Extraction completed!")
                    
                except Exception as e:
                    st.error(f"❌ Extraction failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.extraction_results:
        results = st.session_state.extraction_results
        
        st.divider()
        st.subheader("📊 Merged Data Summary")
        
        # Show merged data info
        merged_count = results['metadata']['total_filled_fields']
        st.info(f"📝 All {results['metadata']['total_pages']} pages have been merged into a single JSON object with **{merged_count} filled fields**. Earlier pages take precedence - later pages only fill in missing values.")
        
        # Show merged JSON
        with st.expander("🔍 View Merged JSON Data", expanded=True):
            st.json(results.get('merged_data', {}))
        
        st.divider()
        st.subheader("📄 Page-by-Page Timing")
        
        # Page timing table with schema info if available
        timing_data = results['page_timings']
        column_config = {
            "page": "Page #",
            "time": st.column_config.NumberColumn("Processing Time (s)", format="%.2f"),
            "filled_fields": "Filled Fields"
        }
        
        # Add schema column if in split schema mode
        if timing_data and 'schema' in timing_data[0]:
            column_config["schema"] = "Schema File"
        
        st.dataframe(
            timing_data,
            column_config=column_config,
            hide_index=True,
            width="stretch"
        )
        
        st.divider()
        
        # Individual page results (for debugging)
        with st.expander("🔧 Debug: Individual Page Results"):
            for page_result in results['pages']:
                schema_info = f" - Schema: {page_result.get('schema', 'N/A')}" if 'schema' in page_result else ""
                with st.expander(f"📄 Page {page_result['page']} - {page_result['filled_fields']} fields ({page_result['processing_time']:.2f}s){schema_info}"):
                    st.json(page_result['data'])

with tab2:
    st.header("✏️ Edit & Approve Data")
    
    if st.session_state.extraction_results:
        # View mode selector
        view_mode = st.radio(
            "View Mode:",
            ["JSON View", "Key-Value Editor"],
            horizontal=True,
            key="view_mode_selector"
        )
        
        if view_mode == "JSON View":
            st.subheader("Merged JSON Data")
            st.info("This is the final merged data from all pages. Later pages have overridden earlier pages where duplicates existed.")
            
            merged_data = st.session_state.extraction_results.get('merged_data', {})
            st.json(merged_data)
        
        else:  # Key-Value Editor
            st.subheader("📝 Editable Fields (Target CSV Format)")
            st.info("Fields are automatically mapped using field_mapping_config.json. Edit values below and approve to export.")
            
            # Get merged data from extraction
            merged_data = st.session_state.extraction_results.get('merged_data', {})
            
            # Apply mapping transformation (only once)
            if st.session_state.edited_data is None:
                with st.spinner("🔄 Mapping extracted data to CSV format..."):
                    mapped_data = apply_mapping(merged_data)
                    st.session_state.edited_data = mapped_data.copy()
            
            # Cache property names in session state
            if 'property_names' not in st.session_state:
                st.session_state.property_names = get_property_names_in_order()
            property_names = st.session_state.property_names
            
            # Show how many fields were pre-populated
            prepopulated_count = sum(1 for v in st.session_state.edited_data.values() if v and str(v).strip())
            total_count = len(property_names)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Fields", total_count)
            with col2:
                st.metric("Pre-populated", prepopulated_count)
            with col3:
                st.metric("Completion", f"{(prepopulated_count/total_count*100):.1f}%")
            
            # Show mapping preview
            with st.expander("🔍 View Mapping Preview (First 10 Fields)"):
                if 'mapping_preview' not in st.session_state:
                    st.session_state.mapping_preview = preview_mapping(merged_data, limit=10)
                st.json(st.session_state.mapping_preview)
            
            # Create editable form with search
            st.divider()
            
            # Search/filter
            search_term = st.text_input("🔍 Search fields", placeholder="Type to filter fields...", key="field_search")
            
            # Filter property names based on search
            if search_term:
                filtered_properties = [p for p in property_names if search_term.lower() in p.lower()]
            else:
                filtered_properties = property_names
            
            st.info(f"Showing {len(filtered_properties)} of {len(property_names)} fields")
            
            # Create editable form (this is the expensive part)
            with st.form("edit_form"):
                # Track edited fields
                edited_fields = {}
                
                # Group fields by category based on field_mapping_config.json order
                # Define category order and field patterns matching the CSV structure
                category_definitions = [
                    ("01. Basic Information", ["Title", "Gender", "First Name", "Last Name", "Middle Name", "Suffix", "Birth Date", "Goes By"]),
                    ("02. Contact Information", ["Social Security Number", "Medicaid Number", "E-mail", "Phone Number"]),
                    ("03. Demographics", ["Race", "Ethnicity", "Tribe", "Class Membership"]),
                    ("04. Residential Address", ["Attention or in care of (Residential", "Residential Street", "Residential City", "Residential State", "Residential Zip", "Residential Country", "Primary Phone (Residential)", "Secondary Phone (Residential)", "Additional Phone (Residential)", "Residential County"]),
                    ("05. Service County", ["Service County State", "Service County"]),
                    ("06. Mailing Address", ["Attention or in care of (Mailing", "Mailing Street", "Mailing City", "Mailing State", "Mailing Zip", "Mailing Country", "Primary Phone (Mailing)", "Secondary Phone (Mailing)", "Additional Phone (Mailing)"]),
                    ("07. Physical Characteristics", ["Height Feet", "Height Inches", "Weight Range", "Hair Color", "Eye Color"]),
                    ("08. Language & Interpreter", ["Interpreter Needed", "Primary Oral Language", "Primary Written Language", "Secondary Oral Language", "Secondary Written Language"]),
                    ("09. Personal Details", ["Religion", "Citizenship", "Marital Status", "Individual's Time Zone", "Living Arrangement"]),
                    ("10. Birth Place", ["Birth Place Street", "Birth Place City", "Birth Place State", "Birth Place Zip", "Birth Place Country"]),
                    ("11. Additional Details", ["Characteristics"]),
                    ("12. Identification", ["ID Type", "ID Number", "Additional ID Type", "Additional ID Number"]),
                    ("13. Disability Information", ["Developmental Disability", "Intellectual Disability"]),
                    ("14. Medical Information", ["Blood Type", "Other Medical Information", "Emergency Orders"]),
                    ("15. Care & Support", ["Adaptive Equipment", "Behavior Management", "Dietary Guidelines", "Eating Guidelines"]),
                    ("16. Communication & Mobility", ["Communication Modality", "Communication Comments", "Mobility", "Supervision"]),
                    ("17. Nutrition & Personal Care", ["Food Texture", "Liquid Consistency", "Toileting Status", "Bathing Status", "Mealtime Status"]),
                    ("18. Administrative", ["Guardian of Self", "Referral Source", "Admission Date", "Program Form ID", "Program Enrollment Date"]),
                ]
                
                # Build categories dict maintaining order
                categories = {cat_name: [] for cat_name, _ in category_definitions}
                categories["19. Other"] = []  # Catch-all for unmatched fields
                
                # Assign each field to its category
                for prop in filtered_properties:
                    assigned = False
                    for cat_name, patterns in category_definitions:
                        if any(pattern in prop for pattern in patterns):
                            categories[cat_name].append(prop)
                            assigned = True
                            break
                    if not assigned:
                        categories["19. Other"].append(prop)
                
                # Remove empty categories
                categories = {k: v for k, v in categories.items() if v}
                
                # Display fields by category
                for category, fields in sorted(categories.items()):
                    # Count populated fields in this category
                    populated_count = sum(
                        1 for field in fields 
                        if st.session_state.edited_data.get(field, "") and str(st.session_state.edited_data.get(field, "")).strip()
                    )
                    
                    # Create header with populated count
                    category_header = f"📁 {category} ({populated_count}/{len(fields)} populated)"
                    
                    with st.expander(category_header, expanded=True):
                        col1, col2 = st.columns(2)
                        for i, field in enumerate(fields):
                            col = col1 if i % 2 == 0 else col2
                            with col:
                                value = st.session_state.edited_data.get(field, "")
                                # Show indicator if field is pre-populated
                                label = f"{'✅' if value and str(value).strip() else '⬜'} {field}"
                                edited_fields[field] = st.text_input(
                                    label,
                                    value=str(value) if value else "",
                                    key=f"edit_{field}",
                                    label_visibility="visible"
                                )
                
                # Submit buttons
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    save_button = st.form_submit_button("💾 Save Changes", width="stretch")
                with col2:
                    reset_button = st.form_submit_button("� Reset to Extracted", width="stretch")
                with col3:
                    approve_button = st.form_submit_button("✅ Approve & Finalize", type="primary", width="stretch")
                
                if save_button:
                    # Update all edited fields
                    for field in property_names:
                        if field in edited_fields:
                            st.session_state.edited_data[field] = edited_fields[field]
                    st.success("✅ Changes saved!")
                    st.rerun()
                
                if reset_button:
                    # Reset to freshly mapped data
                    st.session_state.edited_data = apply_mapping(merged_data)
                    st.success("🔄 Reset to extracted data!")
                    st.rerun()
                
                if approve_button:
                    # Update all edited fields
                    for field in property_names:
                        if field in edited_fields:
                            st.session_state.edited_data[field] = edited_fields[field]
                    st.success("✅ Data approved! Go to Export tab to download CSV.")
                    st.rerun()
    
    else:
        st.info("👆 Upload and extract a PDF first to edit data.")

with tab3:
    st.header("💾 Export Data")
    
    if st.session_state.edited_data:
        st.success("✅ Data ready for export!")
        
        # Get property names in order
        property_names = get_property_names_in_order()
        
        # Preview
        st.subheader("Preview")
        
        # Show filled fields count
        filled_count = sum(1 for v in st.session_state.edited_data.values() if v and str(v).strip())
        total_count = len(property_names)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Filled Fields", f"{filled_count}/{total_count}")
        with col2:
            st.metric("Completion", f"{(filled_count/total_count*100):.1f}%")
        
        # Show data table (only filled fields)
        preview_data = [
            {"Field": k, "Value": v} 
            for k, v in st.session_state.edited_data.items() 
            if v and str(v).strip()  # Only show filled fields
        ]
        
        st.dataframe(preview_data, width="stretch", height=400)
        
        st.divider()
        
        # Append to CSV Section
        st.subheader("📝 Append to Target CSV")
        
        # Get the path to target.csv in parent directory
        target_csv_path = Path(__file__).parent.parent / "target.csv"
        
        # Check if target.csv exists
        if target_csv_path.exists():
            st.info(f"Target file: `{target_csv_path}`")
            
            # Show append status
            if st.session_state.data_appended:
                st.success("✅ This data has already been appended to target.csv")
                st.warning("⚠️ Load a new document to append again")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("Click the button to append this row to the existing target.csv file.")
                
                with col2:
                    if st.button("➕ Append to CSV", type="primary", width="stretch"):
                        try:
                            # Prepare data row in correct order
                            data_row = [st.session_state.edited_data.get(prop, "") for prop in property_names]
                            
                            # Append to CSV file
                            with open(target_csv_path, 'a', newline='', encoding='utf-8') as f:
                                csv_writer = csv.writer(f)
                                csv_writer.writerow(data_row)
                            
                            # Mark as appended
                            st.session_state.data_appended = True
                            st.success("✅ Successfully appended data to target.csv!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Failed to append to CSV: {e}")
        else:
            st.error(f"❌ Target CSV not found at: `{target_csv_path}`")
            st.info("Please ensure target.csv exists in the project directory.")
        
        st.divider()
        
        # Export buttons
        st.subheader("📥 Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Download (in property order)
            import io
            csv_buffer = io.StringIO()
            csv_writer = csv.writer(csv_buffer)
            
            # Write header
            csv_writer.writerow(property_names)
            
            # Write data row in correct order
            data_row = [st.session_state.edited_data.get(prop, "") for prop in property_names]
            csv_writer.writerow(data_row)
            
            csv_string = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_string,
                file_name="extracted_form_data.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col2:
            # JSON Download (full data)
            json_output = {
                "metadata": st.session_state.extraction_results['metadata'],
                "merged_data": st.session_state.extraction_results.get('merged_data', {}),
                "mapped_csv_data": st.session_state.edited_data
            }
            
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_output, indent=2),
                file_name="extracted_form_data.json",
                mime="application/json",
                width="stretch"
            )
        
        with col3:
            # Download mapping config for reference
            mapping_config = load_mapping_config()
            
            st.download_button(
                label="📥 Download Mapping Config",
                data=json.dumps(mapping_config, indent=2),
                file_name="field_mapping_config.json",
                mime="application/json",
                width="stretch"
            )
    
    elif st.session_state.extraction_results:
        st.info("👈 Go to 'Edit & Approve' tab to review and approve the data first.")
    
    else:
        st.info("👆 Upload and extract a PDF first to export data.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>TOT Form Extractor powered by Ollama Vision Models</p>
</div>
""", unsafe_allow_html=True)