import os
import json
import argparse
import logging
import base64
import tempfile
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO
from llm_handler import LLMHandler
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """
You are an expert OCR and form-understanding assistant.

{schema}

You receive a scanned PDF form that contains both printed labels and handwritten responses.
Your job:
1. Extract ONLY the handwritten or user-entered responses.
2. Match each extracted response exactly to the JSON schema provided.
3. If a field is blank, illegible, or missing, return null.
4. Do NOT guess or copy printed text.
5. Return only valid JSON that fits the schema structure exactly.
6. For checkboxes, return the marked options.
7. Normalize dates to YYYY-MM-DD and phone numbers to E.164 if possible.

Return strictly valid JSON. Do not include comments, trailing commas, or extra text.
"""

SCHEMA_MATCH_PROMPT = """
Does this image of a form page match this schema?
1. Prioritize exact matches of any 'enum' fields, then descriptions (quoted from form until ; following is property information), then property names.
2. A correlation of 70% or higher indicates a match.
Reply only 'yes' or 'no'.

Schema:
{schema_text}
"""


def load_schema(schema_path):
    """Load the JSON schema."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_filled_fields(json_data):
    """Count non-null/non-empty fields in JSON data."""
    count = 0
    if isinstance(json_data, dict):
        for value in json_data.values():
            if isinstance(value, (dict, list)):
                count += count_filled_fields(value)
            elif value is not None and value != "" and value != [] and str(value).upper() != "NULL":
                count += 1
    elif isinstance(json_data, list):
        # Count non-empty list items
        for item in json_data:
            if isinstance(item, (dict, list)):
                count += count_filled_fields(item)
            elif item is not None and item != "" and str(item).upper() != "NULL":
                count += 1
    return count


def count_total_fields(schema):
    """Recursively count total fields in a JSON schema (object properties)."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            return sum(count_total_fields(v) for v in schema["properties"].values())
        elif schema.get("type") == "array" and "items" in schema:
            return count_total_fields(schema["items"])
        else:
            return 1
    return 0


def deep_merge(base_dict, update_dict):
    """
    Recursively merge update_dict into base_dict.
    Earlier values (from base_dict) override later values (from update_dict).
    Only adds new keys or fills in missing values from update_dict.
    """
    result = base_dict.copy()
    
    def is_empty_value(val):
        """Check if a value is empty, including NULL string."""
        if val is None or val == "" or val == []:
            return True
        if isinstance(val, str) and val.upper() == "NULL":
            return True
        return False
    
    for key, value in update_dict.items():
        if key in result:
            # Key exists in base
            if isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = deep_merge(result[key], value)
            elif is_empty_value(result[key]):
                # Base value is empty, fill with update value if non-empty
                if not is_empty_value(value):
                    result[key] = value
            # else: base value exists and is non-empty, keep it (don't override)
        else:
            # Key doesn't exist in base, add it if non-empty
            if not is_empty_value(value):
                result[key] = value
    
    return result


def merge_page_extractions(page_results):
    """
    Merge all page extractions into a single JSON object.
    Earlier pages take precedence over later pages.
    """
    merged = {}
    
    for page_result in page_results:
        page_data = page_result.get('data', {})
        merged = deep_merge(merged, page_data)
    
    return merged


def deep_merge_dicts(a, b):
    """
    Deep merge dict b into dict a (used for split schema merging).
    Combines values from multiple pages.
    """
    for key, value in b.items():
        if key in a:
            if isinstance(a[key], dict) and isinstance(value, dict):
                deep_merge_dicts(a[key], value)
            elif isinstance(a[key], list) and isinstance(value, list):
                a[key] = a[key] + [item for item in value if item not in a[key]]
            else:
                if a[key] is None and value is not None:
                    a[key] = value
                elif value is not None:
                    a[key] = value
        else:
            a[key] = value
    return a


def merge_page_results(results):
    """
    Merge all page extractions (for split schema pattern).
    """
    merged = {}
    for page_data in results:
        if not isinstance(page_data, dict):
            continue
        deep_merge_dicts(merged, page_data)
    return merged


def get_all_schema(schema_dir):
    """
    Get all schema files from directory, sorted numerically.
    """
    def extract_num(f):
        stem = Path(f).stem
        num = ''.join(filter(str.isdigit, stem))
        return int(num) if num else 0

    schema_files = sorted(
        Path(schema_dir).glob("schema*.json"),
        key=extract_num
    )
    return schema_files


def match_page_to_schema(llm, page_image, schema_files):
    """
    Match a page image to the most appropriate schema file.
    Returns schema filename or 'none' if no match.
    """
    image_b64 = base64.b64encode(page_image).decode('utf-8')
    
    for schema_file in schema_files:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
        prompt = SCHEMA_MATCH_PROMPT.format(schema_text=schema_text)
        
        for attempt in range(3):
            try:
                response = llm.model.generate(
                    model=llm.model_name,
                    prompt=prompt,
                    images=[image_b64],
                    stream=False,
                    options={
                        "temperature": 0,
                        "top_p": 1,
                        "num_ctx": 10000,
                        "num_predict": 16,
                    }
                )
                answer = response.get('response', '').strip().lower()
                
                if "yes" in answer.lower():
                    logger.info(f"✅ Matched to schema: {schema_file.name}")
                    return schema_file.name
                elif "no" in answer.lower():
                    break
            except Exception as e:
                logger.warning(f"Schema match error for {schema_file.name}: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    break
    
    return "none"


def extract_page_json(llm, page_image, page_num, schema_text):
    """
    Extract JSON data from a single page using its matched schema.
    """
    logger.info(f"Processing page {page_num}...")

    page_prompt = f"""
This is page {page_num} of a multi-page form.
Extract only the handwritten or user-entered responses visible on this page.
Return valid JSON according to the provided schema.
"""

    for attempt in range(3):
        try:
            return llm.generate_json(schema_text, page_prompt, page_image)
        except Exception as e:
            logger.error(f"Error on page {page_num}: {e}")
            if attempt < 2:
                delay = 2 ** attempt
                logger.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logger.error(f"Skipping page {page_num} after repeated errors.")
                raise


def extract_from_pdf(
    pdf_path,
    schema_path,
    output_path=None,
    model_name=None,
    use_split_schema=False,
    schema_dir=None,
    selected_page_indices=None,
    matched_schema_names=None 
):
    """
    Extract structured JSON from a PDF using Ollama vision model.
    
    Args:
        pdf_path: Path to the PDF file
        schema_path: Path to the JSON schema file (used if use_split_schema=False)
        output_path: Optional path to save results
        model_name: Optional model name to use (overrides .env)
        use_split_schema: If True, use per-page schema matching
        schema_dir: Directory containing schema1.json, schema2.json, etc. (required if use_split_schema=True)
        selected_page_indices: Optional list of page indices (0-based) to process. If None, process all pages.
        matched_schema_names: Optional list of schema file names (for split schema mode, overrides auto-matching)
    
    Returns: (results_dict, timing_info)
    """
    start_time = time.time()
    
    # Initialize LLM handler with specified model
    llm = LLMHandler(model_name=model_name)
    
    # Convert PDF to images
    logger.info(f"📄 Converting PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=200)
    logger.info(f"✅ Converted {len(images)} page(s)")
    
    # Filter pages if selected_page_indices is provided
    if selected_page_indices is not None:
        images = [images[i] for i in selected_page_indices if i < len(images)]
        logger.info(f"📌 Processing {len(images)} selected page(s)")
    
    # Process based on schema mode
    if use_split_schema:
        if not schema_dir:
            raise ValueError("schema_dir is required when use_split_schema=True")
        
        logger.info("🔄 Using split schema mode - matching pages to schemas...")
        schema_files = get_all_schema(schema_dir)
        logger.info(f"📁 Found {len(schema_files)} schema files")
        
        matched_pages = []
        matched_schemas = []
        matched_schema_names_list = []

        if matched_schema_names is not None:
            # Use user-provided schema assignments
            logger.info("📝 Using user-confirmed schema assignments for each page.")
            schema_file_map = {f.name: f for f in schema_files}
            for idx, (img, schema_name) in enumerate(zip(images, matched_schema_names), start=1):
                if schema_name not in schema_file_map:
                    logger.warning(f"⚠️ Assigned schema {schema_name} not found. Skipping page {idx}.")
                    continue
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                matched_pages.append(img_bytes)
                matched_schemas.append(schema_file_map[schema_name])
                matched_schema_names_list.append(schema_name)
        else:
            # Original auto-matching logic
            for idx, img in enumerate(images, start=1):
                logger.info(f"\n🔍 Matching page {idx}/{len(images)} to schema...")
                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                schema_name = match_page_to_schema(llm, img_bytes, schema_files)
                if schema_name == "none" or not schema_name:
                    logger.warning(f"⚠️ Page {idx} does not match any schema. Skipping.")
                    continue
                schema_path_matched = Path(schema_dir) / schema_name
                if not schema_path_matched.exists():
                    logger.warning(f"⚠️ Matched schema {schema_name} not found. Skipping page {idx}.")
                    continue
                matched_pages.append(img_bytes)
                matched_schemas.append(schema_path_matched)
                matched_schema_names_list.append(schema_name)
        
        logger.info(f"✅ {len(matched_pages)} pages matched to schemas")
        
        # Extract data from matched pages
        results = []
        page_timings = []
        all_page_data = []
        
        for idx, (img_bytes, schema_path_matched, schema_name) in enumerate(zip(matched_pages, matched_schemas, matched_schema_names_list), start=1):
            page_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Processing Page {idx}/{len(matched_pages)} (Schema: {schema_name})")
            logger.info(f"{'='*60}")
            
            # Load schema for this page
            with open(schema_path_matched, "r", encoding="utf-8") as f:
                schema = json.load(f)
            schema_text = SYSTEM_INSTRUCTIONS.format(
                schema=json.dumps(schema, indent=2, ensure_ascii=False)
            )
            
            # Extract JSON
            page_json = extract_page_json(llm, img_bytes, idx, schema_text)
            
            page_elapsed = time.time() - page_start
            filled_count = count_filled_fields(page_json)
            total_fields = count_total_fields(schema)
            page_timings.append({
                'page': idx,
                'time': page_elapsed,
                'filled_fields': filled_count,
                'total_fields': total_fields,
                'schema': schema_name
            })
            
            logger.info(f"⏱️  Page {idx} processed in {page_elapsed:.2f}s")
            logger.info(f"📊 Filled fields: {filled_count}")
            
            results.append({
                'page': idx,
                'data': page_json,
                'processing_time': page_elapsed,
                'filled_fields': filled_count,
                'schema': schema_name
            })
            
            all_page_data.append(page_json)
        
        total_time = time.time() - start_time
        
        # Merge all pages using split schema merge
        logger.info("\n🔄 Merging data from all pages...")
        merged_data = merge_page_results(all_page_data)
        merged_filled_count = count_filled_fields(merged_data)
        
    else:
        # Original single schema mode
        logger.info("📄 Using single schema mode for all pages...")
        
        # Load schema
        schema = load_schema(schema_path)
        schema_text = json.dumps(schema, indent=2)
        total_fields = count_total_fields(schema)
        
        # Process each page
        results = []
        page_timings = []
        
        for idx, img in enumerate(images, start=1):
            page_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Processing Page {idx}/{len(images)}")
            logger.info(f"{'='*60}")
            
            # Convert image to bytes
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            
            # Prepare page-specific prompt
            page_prompt = f"This is page {idx} of {len(images)}."
            
            # Generate JSON
            page_json = llm.generate_json(schema_text, page_prompt, img_bytes)
            
            page_elapsed = time.time() - page_start
            filled_count = count_filled_fields(page_json)
            
            page_timings.append({
                'page': idx,
                'time': page_elapsed,
                'filled_fields': filled_count,
                'total_fields': total_fields
            })
            
            logger.info(f"⏱️  Page {idx} processed in {page_elapsed:.2f}s")
            logger.info(f"📊 Filled fields: {filled_count}")
            
            results.append({
                'page': idx,
                'data': page_json,
                'processing_time': page_elapsed,
                'filled_fields': filled_count
            })
        
        total_time = time.time() - start_time
        
        # Merge all pages into single JSON (earlier pages take precedence)
        logger.info("\n🔄 Merging data from all pages...")
        merged_data = merge_page_extractions(results)
        merged_filled_count = count_filled_fields(merged_data)
    
    logger.info(f"✅ Merged data contains {merged_filled_count} filled fields")
    
    # Create output structure with metadata
    output_data = {
        'metadata': {
            'pdf_path': str(pdf_path),
            'model_name': llm.model_name,
            'total_pages': len(images),
            'total_time_seconds': round(total_time, 2),
            'average_time_per_page': round(total_time / len(images), 2),
            'total_filled_fields': merged_filled_count,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'schema_mode': 'split' if use_split_schema else 'single'
        },
        'page_timings': page_timings,
        'merged_data': merged_data,  # Single merged JSON
        'pages': results  # Individual page results for debugging
    }
    
    # Save output
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ Results saved to: {output_path}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"⏱️  Total Time: {total_time:.2f}s")
    logger.info(f"📄 Pages Processed: {len(images)}")
    logger.info(f"⚡ Avg Time/Page: {total_time/len(images):.2f}s")
    logger.info(f"📝 Total Filled Fields (Merged): {merged_filled_count}")
    logger.info(f"{'='*60}\n")
    
    return output_data, page_timings


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from TOT forms")
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--schema', required=True, help='Path to JSON schema')
    parser.add_argument('--out', default='output_ollama.json', help='Output JSON path')
    
    args = parser.parse_args()
    
    extract_from_pdf(args.pdf, args.schema, args.out)


if __name__ == "__main__":
    main()
