import base64
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
import tempfile
from llm_handler import LLMHandler

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
You are an expert at matching scanned form pages to their expected schema.

Given the following schema descriptions:
{schema_descriptions}

Analyze the provided page image and determine which schema best matches the content and field descriptions visible on this page.
Return ONLY the schema filename (e.g., "schema3.json") that best fits, or "none" if no schema matches.
"""

def extract_page_json(llm, page_image, page_num, schema_text):
    print(f"Processing page {page_num} ...")

    page_prompt = f"""
This is page {page_num} of a multi-page form.
Extract only the handwritten or user-entered responses visible on this page.
Return valid JSON according to the provided schema.
"""

    for attempt in range(3):
        try:
            return llm.generate_json(schema_text, page_prompt, page_image)
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            if attempt < 2:
                delay = 2 ** attempt
                print(f"Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Skipping page {page_num} after repeated errors.")
                return {}

def deep_merge_dicts(a, b):
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
    merged = {}
    for page_data in results:
        if not isinstance(page_data, dict):
            continue
        deep_merge_dicts(merged, page_data)
    return merged

def get_schema_descriptions(schema_dir):
    schema_files = sorted(Path(schema_dir).glob("schema*.json"))
    descriptions = []
    for schema_file in schema_files:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        desc = []
        for k, v in schema.get("properties", {}).items():
            if isinstance(v, dict) and "description" in v:
                desc.append(f"{k}: {v['description']}")
            else:
                desc.append(k)
        descriptions.append(f"{schema_file.name}: " + "; ".join(desc))
    return schema_files, "\n".join(descriptions)

def match_page_to_schema(llm, page_image, schema_descriptions):
    prompt = SCHEMA_MATCH_PROMPT.format(schema_descriptions=schema_descriptions)
    for attempt in range(3):
        try:
            response = llm.model.generate(
                model=llm.model_name,
                prompt=prompt,
                images=[base64.b64encode(page_image).decode('utf-8')],
                stream=False,
                options={
                    "temperature": 0,
                    "top_p": 1,
                    "num_ctx": 4096,
                    "num_predict": 128,
                }
            )
            schema_name = response.get('response', '').strip().replace('"', '')
            return schema_name
        except Exception as e:
            print(f"Schema match error: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return "none"

def main():
    parser = argparse.ArgumentParser(description="Page-wise LLM OCR with schema output")
    parser.add_argument("--pdf", required=True, help="Path to input filled PDF")
    parser.add_argument("--schema_dir", required=True, help="Directory containing page-wise schema files (schema1.json, schema2.json, ...)")
    parser.add_argument("--out", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    load_dotenv()
    llm = LLMHandler()

    print(f"Converting {args.pdf} to images...")
    pages = convert_from_path(args.pdf, dpi=150)
    print(f"{len(pages)} pages converted.\n")

    schema_files, schema_descriptions = get_schema_descriptions(args.schema_dir)

    matched_pages = []
    matched_schemas = []
    temp_files = [] 
    for i, page in enumerate(pages, start=1):
        print(f"Matching page {i} to schema...")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            page.save(tmp.name, "PNG")
            temp_files.append(tmp.name)
            with open(tmp.name, "rb") as img_file:
                img_bytes = img_file.read()
            schema_name = match_page_to_schema(llm, img_bytes, schema_descriptions)
            if schema_name == "none":
                print(f"Page {i} does not match any schema. Skipping.")
                continue
            schema_path = Path(args.schema_dir) / schema_name
            if not schema_path.exists():
                print(f"Matched schema {schema_name} not found. Skipping page {i}.")
                continue
            matched_pages.append(img_bytes)
            matched_schemas.append(schema_path)

    print(f"{len(matched_pages)} pages matched to schemas.\n")

    all_page_data = []
    for i, (img_bytes, schema_path) in enumerate(zip(matched_pages, matched_schemas), start=1):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema_text = SYSTEM_INSTRUCTIONS.format(
            schema=json.dumps(schema, indent=2, ensure_ascii=False)
        )
        page_json = extract_page_json(llm, img_bytes, i, schema_text)
        all_page_data.append(page_json)

    final_json = merge_page_results(all_page_data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    for temp_path in temp_files:
        try:
            Path(temp_path).unlink()
        except Exception as e:
            print(f"Warning: Could not delete temp file {temp_path}: {e}")

    print(f"\nExtraction complete! Combined JSON saved to {out_path}")

if __name__ == "__main__":
    main()
