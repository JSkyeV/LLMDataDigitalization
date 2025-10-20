# Handwriting Form Extraction Framework

Extract **handwritten or user-entered data** from scanned PDF forms using any **LLM model** in a fully modular and customizable framework.

This project supports both:
- A **command-line tool (`ocr_extractor.py`)** for automated extraction  
- An interactive **Streamlit app (`app.py`)** for visual exploration

---

## Project Structure

```
│
├── ocr_extractor.py       # CLI tool for page-wise PDF extraction
├── app_updated.py         # Streamlit dashboard for interactive form extraction
├── llm_handler.py         # Generic LLM configuration handler
├── ocr_schema.json        # JSON schema defining structure of the empty form
├── requirements.txt       # All dependencies
└── .env                   # Model and API configuration file
```

---

## 1. Installation

Step 1: Clone the repository

Step 2: Create and activate a virtual environment

Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Environment Configuration

All API keys and model details are managed through a `.env` file in the project root.

Example `.env`:
```bash
# .env
LLM_MODEL_NAME=gemini-2.0-flash
LLM_API_KEY_ENV=your_actual_api_key_here
```

You can swap models by simply changing these variables

---

## 3. Configure Your Model (in `llm_handler.py`)

This framework is **provider-agnostic** — no hardcoded defaults.  
You must manually import and initialize your chosen LLM in the **USER CONFIGURATION SECTION** inside `llm_handler.py`.

### Example: Google Gemini
```python
# USER CONFIGURATION SECTION
import google.generativeai as genai

genai.configure(api_key=self.api_key)
self.model = genai.GenerativeModel(self.model_name)
```

### Example: OpenAI GPT-4o
```python
# USER CONFIGURATION SECTION
import openai
openai.api_key = self.api_key
self.model = openai
```

Please also remember to change the corresponding APIs for the model that you choose.

This lets you test any model without changing the extraction logic.

---

## 4. Running the CLI Extractor

Use the CLI for bulk or automated runs:

```bash
python3 ocr_extractor.py \
  --pdf input_file.pdf \
  --schema ocr_schema.json \
  --out output_file.json
```

---

## 5. Running the Streamlit App

For an interactive UI:
```bash
streamlit run app.py
```

Then open the local link (usually http://localhost:8501) in your browser.

The app allows you to:
- Upload scanned forms (PDFs)
- Automatically apply the internal schema (`ocr_schema.json`)
- Preview and download structured JSON results

---

## 6. Output Format

All output strictly follows your defined schema (`ocr_schema.json`):
- Only handwritten or user-entered responses are extracted.
- Blank or illegible fields → `null`
- Checkboxes → extracted as selected options
- Dates → normalized to `YYYY-MM-DD`
- Phone numbers → normalized to E.164 when possible

---


## 7. Switching Between Models

To change LLMs:
1. Edit `.env` → update `LLM_MODEL_NAME`, `LLM_API_KEY_ENV`, and key.
2. Update import/config section in `llm_handler.py`.
3. Run the same `ocr_extractor.py` or `streamlit run app.py`.

No other changes needed — the framework adapts automatically.

---

## 8. Author & Credits

**Author:** Akhil Immadi  
**Purpose:** Research framework for evaluating LLM-based handwriting extraction accuracy.

---

## 9. License

This project is open for educational and research purposes.  
Attribution is appreciated when used in derivative or academic work.

---
