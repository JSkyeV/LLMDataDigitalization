import base64
import json
import time
from pathlib import Path
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
3. The page index is {page_index}, currently checking {current_schema_num}, and the previous schema was matched to {prev_schema_num}.
4. If {current_schema_num} is equal to {prev_schema_num}, respond 'no' to avoid duplicate schema usage.

Schema:
{schema_text}

Respond with only "yes" if the page matches this schema, or "no" if it does not. Do not include any explanation or extra text.
"""

PAGE_PROMPT = """
This is page {page_num} of a multi-page form.
Extract only the handwritten or user-entered responses visible on this page.
Return valid JSON according to the provided schema.
"""

def extract_page_json(llm, page_image, page_num, schema_text):
    cprint(f"Processing page {page_num} ...", "cyan")

    page_prompt = PAGE_PROMPT.format(page_num=page_num)
    try:
        return llm.generate_json(schema_text, page_prompt, page_image)
    except Exception as e:
        cprint(f"Error on page {page_num}: {e}", "red")
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

def match_page_to_schema(llm, page_image, schema_files, page_index=1, prev_schema_num=None):
    """
    For each schema, ask the SCHEMA_MATCH_PROMPT with the page image.
    If 'yes', return that schema filename. If all 'no', return 'none'.
    """
    image_b64 = base64.b64encode(page_image).decode('utf-8')
    for idx, schema_file in enumerate(schema_files):
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema_text = json.dumps(schema, indent=2, ensure_ascii=False)
        stem = Path(schema_file).stem
        current_schema_num = ''.join(filter(str.isdigit, stem)) or "unknown"
        prompt = SCHEMA_MATCH_PROMPT.format(
            schema_text=schema_text,
            page_index=page_index,
            prev_schema_num=prev_schema_num if prev_schema_num is not None else "none",
            current_schema_num=current_schema_num
        )
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
