# llm_config.py
import os
import json
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Data class for model configuration"""
    id: str
    name: str
    description: str
    provider: str  # 'gemini' or 'groq'
    max_tokens: int = 4000
    temperature: float = 0.7


# ─── Available Models ────────────────────────────────────────────────────────

GEMINI_MODELS = {
    "gemini-2.5-flash": ModelConfig(
        id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        description="Fast, efficient – recommended",
        provider="gemini",
    ),
    "gemini-2.5-flash-lite": ModelConfig(
        id="gemini-2.5-flash-lite",
        name="Gemini 2.5 Flash Lite",
        description="Lightning fast, lower cost",
        provider="gemini",
    ),
    "gemini-2.0-flash": ModelConfig(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        description="Balanced performance",
        provider="gemini",
    ),
    "gemini-2.0-flash-lite": ModelConfig(
        id="gemini-2.0-flash-lite",
        name="Gemini 2.0 Flash Lite",
        description="Very fast, efficient",
        provider="gemini",
    ),
    "gemma-3-27b-it": ModelConfig(
        id="gemma-3-27b-it",
        name="Gemma 3 27B",
        description="Open model, good for technical content",
        provider="gemini",
    ),
    "gemma-3-12b-it": ModelConfig(
        id="gemma-3-12b-it",
        name="Gemma 3 12B",
        description="Smaller, faster open model",
        provider="gemini",
    ),
}

GROQ_MODELS = {
    "llama-3.3-70b-versatile": ModelConfig(
        id="llama-3.3-70b-versatile",
        name="Llama 3.3 70B",
        description="Groq – powerful, versatile",
        provider="groq",
    ),
    "llama-3.1-8b-instant": ModelConfig(
        id="llama-3.1-8b-instant",
        name="Llama 3.1 8B Instant",
        description="Groq – ultra-fast, lightweight",
        provider="groq",
    ),
    "mixtral-8x7b-32768": ModelConfig(
        id="mixtral-8x7b-32768",
        name="Mixtral 8x7B",
        description="Groq – strong instruction following",
        provider="groq",
    ),
    "gemma2-9b-it": ModelConfig(
        id="gemma2-9b-it",
        name="Gemma 2 9B (Groq)",
        description="Groq – efficient open model",
        provider="groq",
    ),
}

ALL_MODELS = {**GEMINI_MODELS, **GROQ_MODELS}


# ─── Shared MCQ Helpers ───────────────────────────────────────────────────────

def _create_prompt(topic: str, num_questions: int, difficulty: str, num_choices: int) -> str:
    letters = ["A", "B", "C", "D", "E"][:num_choices]
    letters_str = ", ".join(letters)
    options_template = "\n".join(f'        "{l}": "Option {l} text"' for l in letters)

    return f"""Generate exactly {num_questions} multiple-choice questions on the topic: "{topic}".

CRITICAL REQUIREMENTS:
1. Difficulty level: {difficulty}
2. Each question must have exactly {num_choices} options ({letters_str})
3. Include exactly one correct answer per question
4. Provide a clear, concise, FACTUALLY ACCURATE explanation
5. Output format must be a valid JSON array – nothing else
6. Questions must be EDUCATIONALLY SOUND and FACTUALLY CORRECT

JSON FORMAT (STRICT):
[
  {{
    "question": "Question text here?",
    "options": {{
{options_template}
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation why this is correct"
  }}
]

IMPORTANT:
- Test understanding, not just memorisation
- Make distractors plausible but clearly wrong
- Cover different aspects of the topic
- Return ONLY the JSON array – no markdown, no commentary"""


def _parse_response(response_text: str) -> List[Dict]:
    """Parse API response to extract MCQ list."""
    cleaned = response_text.strip()

    # Strip markdown fences
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try to isolate JSON array
    start = cleaned.find("[")
    end = cleaned.rfind("]") + 1
    if start != -1 and end > 0:
        try:
            parsed = json.loads(cleaned[start:end])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("questions", "mcqs", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    return []


def _validate_mcq(mcq: Dict, num_choices: int) -> Optional[Dict]:
    """Validate and normalise a single MCQ dict."""
    if not isinstance(mcq, dict):
        return None

    question = str(mcq.get("question", "")).strip()
    if len(question) < 5:
        return None

    options = mcq.get("options")
    if not isinstance(options, dict):
        return None

    cleaned_options = {
        k.upper(): str(v).strip()
        for k, v in options.items()
        if k.upper() in "ABCDE" and str(v).strip()
    }
    if len(cleaned_options) < 2:
        return None

    correct = str(mcq.get("correct_answer", "")).strip().upper()
    if correct not in cleaned_options:
        correct = list(cleaned_options.keys())[0]

    explanation = str(mcq.get("explanation", "")).strip() or "No explanation provided."

    return {
        "question": question,
        "options": cleaned_options,
        "correct_answer": correct,
        "explanation": explanation,
    }


# ─── Gemini Client (REST) ─────────────────────────────────────────────────────

class GeminiClient:
    """Google Gemini API client using plain HTTP requests (no SDK required)."""

    AVAILABLE_MODELS = GEMINI_MODELS
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str = None, model_id: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set.\n\n"
                "Add it to your .env file:\n"
                "  GOOGLE_API_KEY=your_key_here\n\n"
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self.set_model(model_id)

    def set_model(self, model_id: str):
        self.model = self.AVAILABLE_MODELS.get(model_id, self.AVAILABLE_MODELS["gemini-2.5-flash"])
        self.model_id = self.model.id

    def get_model_info(self) -> Dict:
        return {"id": self.model_id, "name": self.model.name, "description": self.model.description}

    def get_available_models(self) -> List[Dict]:
        return [{"id": m.id, "name": m.name, "description": m.description} for m in self.AVAILABLE_MODELS.values()]

    def generate_mcqs(self, topic: str, num_questions: int = 5,
                      difficulty: str = "medium", num_choices: int = 4) -> List[Dict]:
        prompt = _create_prompt(topic, num_questions, difficulty, num_choices)
        system_instruction = (
            "You are an expert educational content creator. "
            "You ONLY output valid JSON arrays. Never add any text outside the JSON structure."
        )
        url = f"{self.BASE_URL}/{self.model_id}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.model.temperature,
                "maxOutputTokens": self.model.max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Network error. Check your internet connection.")

        if resp.status_code == 429:
            raise Exception("Rate limit exceeded. Please wait a moment and try again.")
        if resp.status_code == 401 or resp.status_code == 403:
            raise Exception("Invalid or unauthorised Google API key. Check GOOGLE_API_KEY.")
        if not resp.ok:
            try:
                err = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err = resp.text
            raise Exception(f"Gemini API error ({resp.status_code}): {err}")

        data = resp.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise Exception("Unexpected response structure from Gemini API.")

        mcqs = _parse_response(text)
        validated = [_validate_mcq(m, num_choices) for m in mcqs[:num_questions]]
        return [m for m in validated if m]


# ─── Groq Client (REST) ───────────────────────────────────────────────────────

class GroqClient:
    """Groq API client using plain HTTP requests (no SDK required)."""

    AVAILABLE_MODELS = GROQ_MODELS
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str = None, model_id: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set.\n\n"
                "Add it to your .env file:\n"
                "  GROQ_API_KEY=your_key_here\n\n"
                "Get a free key at https://console.groq.com/keys"
            )
        self.set_model(model_id)

    def set_model(self, model_id: str):
        self.model = self.AVAILABLE_MODELS.get(model_id, list(self.AVAILABLE_MODELS.values())[0])
        self.model_id = self.model.id

    def get_model_info(self) -> Dict:
        return {"id": self.model_id, "name": self.model.name, "description": self.model.description}

    def get_available_models(self) -> List[Dict]:
        return [{"id": m.id, "name": m.name, "description": m.description} for m in self.AVAILABLE_MODELS.values()]

    def generate_mcqs(self, topic: str, num_questions: int = 5,
                      difficulty: str = "medium", num_choices: int = 4) -> List[Dict]:
        prompt = _create_prompt(topic, num_questions, difficulty, num_choices)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert educational content creator. "
                        "You ONLY output valid JSON arrays. Never add any text outside the JSON structure."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
        }

        try:
            resp = requests.post(self.BASE_URL, json=payload, headers=headers, timeout=60)
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            raise Exception("Network error. Check your internet connection.")

        if resp.status_code == 429:
            raise Exception("Rate limit exceeded. Please wait a moment and try again.")
        if resp.status_code in (401, 403):
            raise Exception("Invalid or unauthorised Groq API key. Check GROQ_API_KEY.")
        if not resp.ok:
            try:
                err = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                err = resp.text
            raise Exception(f"Groq API error ({resp.status_code}): {err}")

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise Exception("Unexpected response structure from Groq API.")

        mcqs = _parse_response(text)
        validated = [_validate_mcq(m, num_choices) for m in mcqs[:num_questions]]
        return [m for m in validated if m]


# ─── Multi-provider Client ────────────────────────────────────────────────────

class MultiLLMClient:
    """
    Unified client that wraps both GeminiClient and GroqClient.
    Automatically picks the right backend based on the selected model.
    """

    def __init__(self):
        self._gemini: Optional[GeminiClient] = None
        self._groq: Optional[GroqClient] = None
        self._active_model_id: Optional[str] = None
        self._errors: Dict[str, str] = {}

        # Try to initialise each provider
        try:
            self._gemini = GeminiClient()
        except ValueError as e:
            self._errors["gemini"] = str(e)

        try:
            self._groq = GroqClient()
        except ValueError as e:
            self._errors["groq"] = str(e)

        if not self._gemini and not self._groq:
            raise ValueError(
                "No API keys found.\n\n"
                "Please set at least one of:\n"
                "  GOOGLE_API_KEY  (Gemini models)\n"
                "  GROQ_API_KEY    (Groq / Llama models)\n\n"
                "in your .env file."
            )

        # Pick a sensible default model
        if self._gemini:
            self._active_model_id = "gemini-2.5-flash"
            self._active_client = self._gemini
        else:
            self._active_model_id = list(GROQ_MODELS.keys())[0]
            self._active_client = self._groq

    # ── Public API ─────────────────────────────────────────────────────────

    def set_model(self, model_id: str):
        model = ALL_MODELS.get(model_id)
        if model is None:
            return  # ignore unknown models

        if model.provider == "gemini":
            if self._gemini is None:
                raise ValueError("Gemini API key not configured. Set GOOGLE_API_KEY.")
            self._gemini.set_model(model_id)
            self._active_client = self._gemini
        else:
            if self._groq is None:
                raise ValueError("Groq API key not configured. Set GROQ_API_KEY.")
            self._groq.set_model(model_id)
            self._active_client = self._groq

        self._active_model_id = model_id

    def get_model_info(self) -> Dict:
        return self._active_client.get_model_info()

    def get_available_models(self) -> List[Dict]:
        """Return models for providers that have valid API keys."""
        models = []
        if self._gemini:
            models += self._gemini.get_available_models()
        if self._groq:
            models += self._groq.get_available_models()
        return models

    def generate_mcqs(self, topic: str, num_questions: int = 5,
                      difficulty: str = "medium", num_choices: int = 4) -> List[Dict]:
        return self._active_client.generate_mcqs(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            num_choices=num_choices,
        )

    def provider_status(self) -> Dict[str, str]:
        """Return availability status for each provider."""
        status = {}
        status["gemini"] = "✅ Available" if self._gemini else f"❌ {self._errors.get('gemini', 'Not configured')}"
        status["groq"] = "✅ Available" if self._groq else f"❌ {self._errors.get('groq', 'Not configured')}"
        return status


# ─── .env Template ───────────────────────────────────────────────────────────

def create_env_template() -> str:
    env_content = """# LLM API Keys
# Get your Google key at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Get your Groq key at: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
"""
    with open(".env", "w") as f:
        f.write(env_content)
    return env_content


# ─── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧪 Testing MultiLLMClient...")
    try:
        client = MultiLLMClient()
        print("Status:", client.provider_status())
        mcqs = client.generate_mcqs("Python basics", num_questions=1, difficulty="easy")
        if mcqs:
            print(f"✅ Generated question: {mcqs[0]['question'][:60]}...")
        else:
            print("❌ No questions generated")
    except Exception as e:
        print(f"❌ Error: {e}")