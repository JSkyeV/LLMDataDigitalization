# To use DoNUT models
    # run on python 3.8 - 3.10 (3.10.11 used) venv and install the following dependencies:
        
        # pip install numpy==1.24.4
        # pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cpu
        # pip install pytorch-lightning==1.9.0 torchmetrics==0.11.4
        # pip install transformers==4.26.1 timm==0.6.13
        # pip install git+https://github.com/clovaai/donut.git 
    # alter to run on GPU (optional):
        # install https://developer.nvidia.com/cuda-toolkit matching your NVIDIA driver
        # for NNIDIA GPU run change cu number to your CUDA: pip install torch==2.9.0+cu130 torchvision==0.24.0+cu130 torchaudio==2.9.0+cu130 --index-url https://download.pytorch.org/whl/cu130
    # run to check versions: pip show donut-python torch torchvision torchaudio
    # set max_length: int = 10000, # 1536 -> 10000 in donut model.py line 358
# To use Pix2Struct models
    # run on python 3.8 - 3.11 venv and install the following dependencies:
        # from 4.26.1 -> 4.29.0: pip install transformers>=4.29.0
        # pip install accelerate
        # pip install bitsandbytes

import os
import json
from dotenv import load_dotenv
from json_repair import repair_json
import google.generativeai as genai

import io
from PIL import Image
# DoNUT imports
if (os.getenv("LLM_MODEL_NAME") and os.getenv("LLM_MODEL_NAME").lower() in ["naver-clova-ix/donut-base"]):
    import torch
    from donut import DonutModel
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        BitsAndBytesConfig = None
# Pix2Struct imports
if (os.getenv("LLM_MODEL_NAME") and os.getenv("LLM_MODEL_NAME").lower() in ["google/pix2struct-base"]):
    import torch
    try:
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
        from transformers import BitsAndBytesConfig
    except ImportError:
        Pix2StructForConditionalGeneration = None
        Pix2StructProcessor = None
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
        api_key_env = os.getenv("LLM_API_KEY_ENV")
        self.api_key = api_key_env       #os.getenv(api_key_env) if api_key_env else None

        if not self.model_name or not self.api_key:
            raise RuntimeError(
                "Missing required environment variables: "
                "LLM_MODEL_NAME and/or LLM_API_KEY_ENV."
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
        # DoNUT integration
        if self.model_name.lower() in ["naver-clova-ix/donut-base"]:
            if DonutModel is None:
                raise ImportError("DonutModel not installed. Please install 'donut'.")
            donut_model_path = os.getenv("DONUT_MODEL_PATH", self.model_name.lower())
            print("IS CUDA AVAILABLE: ", torch.cuda.is_available())
            use_bit = os.getenv("LLM_USE_BIT", "0")
            if use_bit == "8":
                if BitsAndBytesConfig is None:
                    raise ImportError("BitsAndBytesConfig not available. Please install 'transformers' >= 4.29.0 and 'bitsandbytes'.")
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = DonutModel.from_pretrained(
                    donut_model_path,
                    max_length=10000,
                    device_map="auto",
                    quantization_config=quant_config
                )
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = DonutModel.from_pretrained(donut_model_path, max_length=10000)
                self.model.to(self.device)
                if self.device.type == "cuda" and use_bit == "16":
                    self.model.half()
            # Check for meta tensors
            meta_params = [n for n, p in self.model.named_parameters() if p.device.type == "meta"]
            meta_buffers = [n for n, b in self.model.named_buffers() if b.device.type == "meta"]
            if meta_params or meta_buffers:
                raise RuntimeError(
                    "Model contains meta tensors (lazy init). Reload the model without init_empty_weights/device_map so weights are materialized."
                )
            self.model.eval()
            self.provider = "donut"
        elif self.model_name.lower() in ["google/pix2struct-base"]: # could try pix2struct-docvqa
            if Pix2StructForConditionalGeneration is None or Pix2StructProcessor is None:
                raise ImportError("transformers>=4.29.0 required for Pix2Struct.")
            pix2struct_model_path = os.getenv("PIX2STRUCT_MODEL_PATH", self.model_name)
            self.processor = Pix2StructProcessor.from_pretrained(pix2struct_model_path)
            # Precision/quantization selection via LLM_USE_BIT
            use_bit = os.getenv("LLM_USE_BIT", "0")
            if use_bit == "8":
                try:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = Pix2StructForConditionalGeneration.from_pretrained(
                        pix2struct_model_path,
                        device_map="auto",
                        quantization_config=quant_config
                    )
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load Pix2Struct in 8-bit: {e}"
                    )
            else:
                self.model = Pix2StructForConditionalGeneration.from_pretrained(pix2struct_model_path)
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                if self.device.type == "cuda" and use_bit == "16":
                    self.model.half()
            self.provider = "pix2struct"
        else:
            raise RuntimeError(
                f"Unsupported model '{self.model_name}'. Only 'donut-base' and 'pix2struct-docvqa' are supported."
            )
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

    def generate_json(self, schema_text, page_prompt, image_bytes):
        if getattr(self, "provider", None) == "donut":
            try:
                # Convert image bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                prompt = schema_text + " " + page_prompt
                # DoNUT expects a prompt and image
                use_bit = os.getenv("LLM_USE_BIT", "0")
                with torch.no_grad():
                    # No need to cast image, only model precision matters
                    output = self.model.inference(image, prompt)
                # Output is usually a dict or string
                if isinstance(output, str):
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        start, end = output.find("{"), output.rfind("}")
                        if start != -1 and end != -1:
                            candidate = output[start:end + 1]
                            return json.loads(repair_json(candidate))
                        raise
                return output
            except Exception as e:
                raise RuntimeError(f"DoNUT generation failed: {e}")
        elif getattr(self, "provider", None) == "pix2struct":
            try:
                # Pix2Struct expects PIL image and a text prompt
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                prompt = schema_text + " " + page_prompt
                inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                # Move inputs to device
                use_bit = os.getenv("LLM_USE_BIT", "0")
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)
                    # Only cast floating point tensors to half if using FP16
                    if self.model.dtype == torch.float16 and use_bit == "16" and torch.is_floating_point(inputs[k]):
                        inputs[k] = inputs[k].half()
                outputs = self.model.generate(**inputs, max_new_tokens=512)
                result = self.processor.decode(outputs[0], skip_special_tokens=True)
                # Try to parse JSON from result
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    start, end = result.find("{"), result.rfind("}")
                    if start != -1 and end != -1:
                        candidate = result[start:end + 1]
                        return json.loads(repair_json(candidate))
                    raise
            except Exception as e:
                raise RuntimeError(f"Pix2Struct generation failed: {e}")
        else:
            # Gemini logic
            try:
                response = self.model.generate_content(
                    [
                        {"role": "user", "parts": [
                            {"text": schema_text},
                            {"text": page_prompt},
                            {"mime_type": "image/png", "data": image_bytes}
                        ]}
                    ],
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "response_mime_type": "application/json",
                    },
                    request_options={"timeout": 180}
                )

                text_output = getattr(response, "text", str(response))
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
