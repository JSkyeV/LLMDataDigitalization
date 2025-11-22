import os
import json
import argparse
import logging
from pathlib import Path
from pdf2image import convert_from_path
from io import BytesIO
from llm_handler import LLMHandler
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def extract_from_pdf(pdf_path, schema_path, output_path=None, model_name=None):
    """
    Extract structured JSON from a PDF using Ollama vision model.
    
    Args:
        pdf_path: Path to the PDF file
        schema_path: Path to the JSON schema file
        output_path: Optional path to save results
        model_name: Optional model name to use (overrides .env)
    
    Returns: (results_dict, timing_info)
    """
    start_time = time.time()
    
    # Load schema
    schema = load_schema(schema_path)
    schema_text = json.dumps(schema, indent=2)
    
    # Initialize LLM handler with specified model
    llm = LLMHandler(model_name=model_name)
    
    # Convert PDF to images
    logger.info(f"📄 Converting PDF: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=200)
    logger.info(f"✅ Converted {len(images)} page(s)")
    
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
            'filled_fields': filled_count
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
    
    # Merge all pages into single JSON (later pages override earlier)
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
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
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