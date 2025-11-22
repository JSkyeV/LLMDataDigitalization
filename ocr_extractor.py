import base64
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from pdf2image import convert_from_path
import tempfile
from llm_handler import LLMHandler
from termcolor import cprint

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

def extract_page_json(llm, page_image, page_num, schema_text):
    cprint(f"Processing page {page_num} ...", "cyan")

    page_prompt = f"""
This is page {page_num} of a multi-page form.
Extract only the handwritten or user-entered responses visible on this page.
Return valid JSON according to the provided schema.
"""

    for attempt in range(3):
        try:
            return llm.generate_json(schema_text, page_prompt, page_image)
        except Exception as e:
            cprint(f"Error on page {page_num}: {e}", "red")
            if attempt < 2:
                delay = 2 ** attempt
                cprint(f"Retrying in {delay}s...", "yellow")
                time.sleep(delay)
            else:
                cprint(f"Skipping page {page_num} after repeated errors.", "red")
                raise

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

def get_all_schema(schema_dir):
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
    For each schema, ask the SCHEMA_MATCH_PROMPT with the page image.
    If 'yes', return that schema filename. If all 'no', return 'none'.
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
                if ("yes" in answer.lower()):
                    return schema_file.name
                elif ("no" in answer.lower()):
                    break
            except Exception as e:
                cprint(f"Schema match error for {schema_file.name}: {e}", "red")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    break
    return "none"

def main():
    try:
        parser = argparse.ArgumentParser(description="Page-wise LLM OCR with schema output")
        parser.add_argument("--pdf", required=True, help="Path to input filled PDF")
        parser.add_argument("--schema_dir", required=True, help="Directory containing page-wise schema files (schema1.json, schema2.json, ...)")
        parser.add_argument("--out", required=True, help="Path to output JSON file")
        args = parser.parse_args()

        load_dotenv()
        llm = LLMHandler()

        cprint(f"PDF: {args.pdf}", "green")
        cprint(f"Converting {args.pdf} to images...", "cyan")
        pages = convert_from_path(args.pdf, dpi=150)
        cprint(f"Pages: {pages}", "green")
        cprint(f"{len(pages)} pages converted.\n", "cyan")

        schema_files = get_all_schema(args.schema_dir)

        matched_pages = []
        matched_schemas = []
        temp_files = []
        for i, page in enumerate(pages, start=1):
            cprint(f"Matching page {i} to schema...", "cyan")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                page.save(tmp.name, "PNG")
                temp_files.append(tmp.name)
                with open(tmp.name, "rb") as img_file:
                    img_bytes = img_file.read()
                schema_name = match_page_to_schema(llm, img_bytes, schema_files)
                if schema_name == "none" or not schema_name:
                    cprint(f"Page {i} does not match any schema. Skipping.", "yellow")
                    continue
                schema_path = Path(args.schema_dir) / schema_name
                if not schema_path.exists():
                    cprint(f"Matched schema {schema_name} not found. Skipping page {i}.", "yellow")
                    continue
                matched_pages.append(img_bytes)
                matched_schemas.append(schema_path)

        cprint(f"{len(matched_pages)} pages matched to schemas.\n", "cyan")

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
                cprint(f"Warning: Could not delete temp file {temp_path}: {e}", "yellow")

        cprint(f"\nExtraction complete! Combined JSON saved to {out_path}", "green")
    except Exception as e:
        cprint(f"\nFatal error: {e}", "red")
        raise

if __name__ == "__main__":
    main()
