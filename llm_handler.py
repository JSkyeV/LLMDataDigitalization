import os
import json
import base64
import re
import logging
from io import BytesIO
from dotenv import load_dotenv
from json_repair import repair_json
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMHandler:
    def __init__(self):
        """Initialize the LLM handler to use Ollama."""
        load_dotenv()

        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5vl")
        
        if not self.model_name:
            raise RuntimeError("Missing OLLAMA_MODEL in .env")

        # Verify Ollama is running and model is available
        try:
            # Test connection - ollama.list() returns a dict with 'models' key
            models_response = ollama.list()
            
            # Extract model names - the structure is different than expected
            available_models = []
            if hasattr(models_response, 'models'):
                # If it's an object with .models attribute
                for model in models_response.models:
                    if hasattr(model, 'name'):
                        available_models.append(model.name)
                    elif hasattr(model, 'model'):
                        available_models.append(model.model)
            elif isinstance(models_response, dict) and 'models' in models_response:
                # If it's a dict
                for model in models_response['models']:
                    if isinstance(model, dict):
                        available_models.append(model.get('name') or model.get('model', ''))
                    elif hasattr(model, 'name'):
                        available_models.append(model.name)
                    elif hasattr(model, 'model'):
                        available_models.append(model.model)
            
            logger.info(f"Available Ollama models: {available_models}")
            
            # Check if our model exists (with or without :latest suffix)
            model_base = self.model_name.split(':')[0]
            model_found = any(
                model_base in m or self.model_name in m 
                for m in available_models
            )
            
            if not model_found and available_models:
                logger.warning(f"Model '{self.model_name}' not found. Available: {available_models}")
                logger.info(f"Attempting to pull model '{self.model_name}'...")
                try:
                    ollama.pull(self.model_name)
                    logger.info(f"✅ Successfully pulled {self.model_name}")
                except Exception as pull_error:
                    logger.error(f"Failed to pull model: {pull_error}")
                    raise
            
            logger.info(f"✅ Connected to Ollama at {self.base_url}")
            logger.info(f"✅ Using model: {self.model_name}")
        except Exception as e:
            logger.error(f"Ollama connection error: {e}")
            logger.info("Attempting to proceed anyway - Ollama may still work...")
            # Don't raise - allow it to fail later if there's a real problem

    def _extract_json_from_response(self, text):
        """Extract JSON from Ollama output."""
        logger.info(f"DEBUG: Response length: {len(str(text))} characters")
        logger.info(f"DEBUG: First 500 chars: {str(text)[:500]}")
        
        text_str = str(text)
        
        # Remove markdown code blocks
        text_str = re.sub(r'```json\s*', '', text_str)
        text_str = re.sub(r'```\s*$', '', text_str)
        text_str = text_str.strip()
        
        # Try direct parse
        try:
            parsed = json.loads(text_str)
            logger.info(f"✅ Successfully parsed JSON directly")
            return parsed
        except json.JSONDecodeError as e:
            logger.info(f"⚠️ Direct parse failed at pos {e.pos}")
        
        # Extract JSON object or array
        json_match = re.search(r'[\{\[].*[\}\]]', text_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
            try:
                parsed = json.loads(json_str)
                logger.info(f"✅ Parsed after extraction")
                return parsed
            except json.JSONDecodeError:
                # Try repair
                try:
                    repaired = repair_json(json_str)
                    parsed = json.loads(repaired)
                    logger.info(f"✅ Parsed after repair")
                    return parsed
                except Exception as e_repair:
                    logger.info(f"❌ Repair failed: {e_repair}")
        
        # Return raw text if JSON extraction fails
        logger.info(f"⚠️ Could not parse as JSON. Returning raw text.")
        return {"raw_text": text_str[:5000]}

    def generate_json(self, schema_text, page_prompt, image_bytes):
        """Generate JSON response using Ollama with vision capabilities."""
        try:
            # Encode image to base64 (Ollama expects base64 for images)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Optimized prompt for OCR and form extraction
            full_prompt = f"""You are an expert OCR system. Extract ALL text from this form image.

Schema structure:
{schema_text[:2000]}

EXTRACTION RULES:
1. Read ALL handwritten and printed text on this page
2. Extract COMPLETE values (full names like "John Michael Smith", not just "J")
3. Extract COMPLETE addresses (like "123 Main Street", not just "1")
4. Extract COMPLETE phone numbers (like "555-123-4567", not just "5")
5. Extract COMPLETE dates (like "01/15/1980", not just "1")
6. For checkboxes: only include checked items in arrays
7. If a field is blank/empty, omit it entirely from the JSON
8. Match extracted text to the schema field names

EXAMPLES OF GOOD EXTRACTION:
- Name: "John Michael Doe" (NOT "J" or "JMD")
- Address: "123 Main Street" (NOT "1" or "123")
- Phone: "555-123-4567" (NOT "5")
- Date: "01/15/1980" (NOT "01" or "1980")

{page_prompt}

Return ONLY valid JSON with the extracted data."""

            logger.info(f"🔄 Sending request to Ollama model: {self.model_name}")
            
            # Call Ollama with vision
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': full_prompt,
                        'images': [image_base64]
                    }
                ],
                options={
                    'temperature': 0.2,  # Slightly higher for better text generation
                    'num_predict': 8192,  # Increased max tokens for complete extraction
                }
            )
            
            text_output = response['message']['content']
            
            logger.info(f"📥 Received: {len(str(text_output))} characters")

            # Extract and return JSON
            return self._extract_json_from_response(text_output)

        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")