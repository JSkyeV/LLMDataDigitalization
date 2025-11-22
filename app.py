import streamlit as st
import json
import time
import csv
from pathlib import Path
from ocr_extractor import extract_from_pdf, count_filled_fields
from field_transformer import (
    apply_mapping,
    get_property_names_in_order,
    load_mapping_config,
    preview_mapping
)

st.set_page_config(page_title="TOT Form Extractor", page_icon="📄", layout="wide")

# Initialize session state
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None
if 'edited_data' not in st.session_state:
    st.session_state.edited_data = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "JSON View"
if 'data_appended' not in st.session_state:
    st.session_state.data_appended = False

st.title("📄 TOT Form Data Extractor")
st.markdown("Extract structured data from PDF forms using Ollama Vision Models")

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_pdf = st.file_uploader("Upload PDF Form", type=['pdf'])
    
    # Schema mode selector
    use_split_schema = st.checkbox(
        "Use Split Schema (per-page matching)",
        value=False,
        help="Enable to match each page to different schema files (schema1.json, schema2.json, etc.)"
    )
    
    if use_split_schema:
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
        options=["qwen2.5vl:7b", "qwen3-vl:4b", "qwen3-vl:8b", "minicpm-v:8b", "gemma3:4b"],
        index=0,
        help="Choose the vision model to use for extraction"
    )
    
    st.info(f"🤖 Using: **{model_name}**")
    
    extract_button = st.button("🚀 Extract Data", type="primary", use_container_width=True)

# Main area tabs
tab1, tab2, tab3 = st.tabs(["📊 Extraction", "✏️ Edit & Approve", "💾 Export"])

with tab1:
    st.header("Extraction Results")
    
    if extract_button and not uploaded_pdf:
        st.warning("⚠️ Please upload a PDF file first!")
    
    if extract_button and uploaded_pdf:
        # Save uploaded file temporarily
        pdf_path = Path("temp_upload.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        
        # Run extraction with progress
        with st.spinner("🔄 Processing PDF..."):
            start_time = time.time()
            
            try:
                if use_split_schema:
                    # Split schema mode - match each page to a schema
                    schema_dir = Path(__file__).parent.parent / "schema"
                    
                    if not schema_dir.exists():
                        st.error(f"❌ Schema directory not found: {schema_dir}")
                        st.stop()
                    
                    results, timings = extract_from_pdf(
                        str(pdf_path),
                        schema_path=None,  # Not used in split mode
                        output_path=None,
                        model_name=model_name,
                        use_split_schema=True,
                        schema_dir=str(schema_dir)
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
                        use_split_schema=False
                    )
                
                st.session_state.extraction_results = results
                st.session_state.edited_data = None  # Reset edited data
                st.session_state.data_appended = False  # Reset append flag for new document
                
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
            use_container_width=True
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
            
            # Apply mapping transformation
            if st.session_state.edited_data is None:
                mapped_data = apply_mapping(merged_data)
                st.session_state.edited_data = mapped_data.copy()
                st.success("✅ Data automatically mapped from extracted JSON using field_mapping_config.json")
            
            # Get property names in order
            property_names = get_property_names_in_order()
            
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
                preview = preview_mapping(merged_data, limit=10)
                st.json(preview)
            
            # Create editable form with search
            st.divider()
            
            # Search/filter
            search_term = st.text_input("🔍 Search fields", placeholder="Type to filter fields...")
            
            # Filter property names based on search
            if search_term:
                filtered_properties = [p for p in property_names if search_term.lower() in p.lower()]
            else:
                filtered_properties = property_names
            
            st.info(f"Showing {len(filtered_properties)} of {len(property_names)} fields")
            
            # Create editable form
            with st.form("edit_form"):
                # Track edited fields
                edited_fields = {}
                
                # Group fields dynamically by category (based on first word)
                categories = {}
                for prop in filtered_properties:
                    # Determine category
                    if any(x in prop for x in ["First Name", "Last Name", "Middle Name", "Title", "Gender", "Suffix", "Birth Date", "Goes By", "Social Security", "Medicaid"]):
                        category = "Personal Information"
                    elif any(x in prop for x in ["Race", "Ethnicity", "Tribe", "Class"]):
                        category = "Demographics"
                    elif "Residential" in prop:
                        category = "Residential Address"
                    elif "Mailing" in prop:
                        category = "Mailing Address"
                    elif "Service" in prop:
                        category = "Service Information"
                    elif any(x in prop for x in ["Height", "Weight", "Hair", "Eye"]):
                        category = "Physical Characteristics"
                    elif "Language" in prop or "Interpreter" in prop:
                        category = "Language & Communication"
                    elif "Birth Place" in prop:
                        category = "Birth Place"
                    elif any(x in prop for x in ["Religion", "Citizenship", "Marital", "Living", "Time Zone", "Characteristics"]):
                        category = "Personal Details"
                    elif any(x in prop for x in ["ID Type", "ID Number"]):
                        category = "Identification"
                    elif any(x in prop for x in ["Disability", "Blood", "Medical", "Emergency"]):
                        category = "Medical Information"
                    elif any(x in prop for x in ["Adaptive", "Behavior", "Dietary", "Eating", "Communication", "Mobility", "Supervision", "Food", "Liquid", "Toileting", "Bathing", "Mealtime"]):
                        category = "Care & Support"
                    elif any(x in prop for x in ["Guardian", "Referral", "Admission", "Program"]):
                        category = "Administrative"
                    else:
                        category = "Other"
                    
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(prop)
                
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
                    save_button = st.form_submit_button("💾 Save Changes", use_container_width=True)
                with col2:
                    reset_button = st.form_submit_button("� Reset to Extracted", use_container_width=True)
                with col3:
                    approve_button = st.form_submit_button("✅ Approve & Finalize", type="primary", use_container_width=True)
                
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
        
        st.dataframe(preview_data, use_container_width=True, height=400)
        
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
                    if st.button("➕ Append to CSV", type="primary", use_container_width=True):
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
                use_container_width=True
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
                use_container_width=True
            )
        
        with col3:
            # Download mapping config for reference
            mapping_config = load_mapping_config()
            
            st.download_button(
                label="📥 Download Mapping Config",
                data=json.dumps(mapping_config, indent=2),
                file_name="field_mapping_config.json",
                mime="application/json",
                use_container_width=True
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