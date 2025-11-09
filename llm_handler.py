import base64
import os
import json
from dotenv import load_dotenv
from json_repair import repair_json
import ollama

class LLMHandler:
    def __init__(self):
        """
        Initialize the LLM using environment variables.

        Required in .env:
            LLM_MODEL_NAME       -> name of the model (string)
            LLM_API_KEY_ENV      -> name of the env variable that stores API key
        Optional (user-defined):
            Any other vars needed for your chosen provider (e.g., API base URL)
        """

        load_dotenv()

        self.model_name = os.getenv("LLM_MODEL_NAME")
        # API key retrieval commented out for Ollama usage
        # api_key_env = os.getenv("LLM_API_KEY_ENV")
        # self.api_key = api_key_env       #os.getenv(api_key_env) if api_key_env else None
        self.ollama_host = os.getenv("LLM_OLLAMA_HOST")

        if not self.model_name:
            raise RuntimeError(
                "Missing required environment variables: "
                "LLM_MODEL_NAME."
            )

        # -------------------------------------------------------------
        # 🔧 USER CONFIGURATION SECTION
        # -------------------------------------------------------------
        # Import and initialize your model explicitly here.
        #
        # Example for Google Gemini:
        # genai.configure(api_key=self.api_key)
        # self.model = genai.GenerativeModel(self.model_name)
        #
        # Example for OpenAI:
        #   import openai
        #   openai.api_key = self.api_key
        #   self.model = openai
        #
        # Example for Anthropic Claude:
        #   from anthropic import Anthropic
        #   self.model = Anthropic(api_key=self.api_key)
        #
        # -------------------------------------------------------------
        #     Developers must uncomment and modify this section
        #     according to their chosen provider.
        # -------------------------------------------------------------
        
        # Utilizing Ollama for a call to a locally-running LLM
        self.model = ollama.Client(host=self.ollama_host) if self.ollama_host else ollama.Client()

    def generate_json(self, schema_text, page_prompt, image_bytes):
        try:
            # Convert image bytes to base64 for Ollama
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Combine the prompts
            combined_prompt = f"{schema_text}\n\n{page_prompt}"

            # print(combined_prompt)  # Debug: print the combined prompt
        
            # Call Ollama's generate method
            response = self.model.generate(
                model=self.model_name,
                prompt=combined_prompt,
                images=[image_base64],
                stream=False,
                options={
                    "temperature": 0,
                    "top_p": 1,
                    "num_ctx": 10000,       # Large context for your big schema
                    "num_predict": 3000,    # Enough tokens for detailed JSON output
                }
            )

            # Extract response text
            text_output = response.get('response', '')

            try:
                return json.loads(text_output)
            except json.JSONDecodeError:
                start, end = text_output.find("{"), text_output.rfind("}")
                if start != -1 and end != -1:
                    candidate = text_output[start:end + 1]
                    return json.loads(repair_json(candidate))
                raise

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")
