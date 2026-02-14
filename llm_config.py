# llm_config.py
import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from google import genai
from google.genai import types


@dataclass
class GeminiModel:
    """Data class for Gemini model configuration"""
    id: str
    name: str
    description: str
    max_tokens: int = 4000
    temperature: float = 0.7


class GeminiClient:
    """Gemini API client for MCQ generation using google.genai SDK"""
    
    # Available free models on Gemini
    AVAILABLE_MODELS = {
        "gemini-2.5-flash": GeminiModel(
            id="gemini-2.5-flash",
            name="Gemini 2.5 Flash",
            description="Fast, efficient, good for most tasks"
        ),
        "gemini-2.5-flash-lite": GeminiModel(
            id="gemini-2.5-flash-lite",
            name="Gemini 2.5 Flash Lite",
            description="Lightning fast, lower cost"
        ),
        "gemini-2.0-flash": GeminiModel(
            id="gemini-2.0-flash",
            name="Gemini 2.0 Flash",
            description="Balanced performance"
        ),
        "gemini-2.0-flash-lite": GeminiModel(
            id="gemini-2.0-flash-lite",
            name="Gemini 2.0 Flash Lite",
            description="Very fast, efficient"
        ),
        "gemma-3-27b-it": GeminiModel(
            id="gemma-3-27b-it",
            name="Gemma 3 27B",
            description="Open model, good for technical content"
        ),
        "gemma-3-12b-it": GeminiModel(
            id="gemma-3-12b-it",
            name="Gemma 3 12B",
            description="Smaller, faster open model"
        )
    }
    
    def __init__(self, api_key: str = None, model_id: str = "gemini-2.5-flash"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google AI Studio API key (optional, will check env var)
            model_id: Model ID to use
        """
        # Try to get API key from parameter, then environment variable
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable is not set.\n\n"
                "Please create a `.env` file in the same directory with this content:\n"
                "GOOGLE_API_KEY=your_google_api_key_here\n\n"
                "Get your API key from: https://aistudio.google.com/app/apikey"
            )
        
        # Initialize Gemini client
        try:
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini client: {str(e)}")
        
        # Set model
        self.set_model(model_id)
    
    def set_model(self, model_id: str):
        """Set the model to use"""
        if model_id not in self.AVAILABLE_MODELS:
            model_id = "gemini-2.5-flash"  # Default fallback
        
        self.model = self.AVAILABLE_MODELS[model_id]
        self.model_id = model_id
    
    def get_model_info(self) -> Dict:
        """Get current model information"""
        return {
            "id": self.model_id,
            "name": self.model.name,
            "description": self.model.description
        }
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models for display"""
        return [
            {
                "id": model_id,
                "name": model.name,
                "description": model.description
            }
            for model_id, model in self.AVAILABLE_MODELS.items()
        ]
    
    def generate_mcqs(self, 
                     topic: str, 
                     num_questions: int = 5, 
                     difficulty: str = "medium",
                     num_choices: int = 4) -> List[Dict]:
        """
        Generate MCQs based on a given topic using Gemini
        
        Args:
            topic: Topic to generate questions about
            num_questions: Number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            num_choices: Number of options per question
            
        Returns:
            List of MCQ dictionaries
        """
        prompt = self._create_prompt(topic, num_questions, difficulty, num_choices)
        
        try:
            response = self._call_api(prompt)
            mcqs = self._parse_response(response)
            
            # Validate and clean the MCQs
            validated_mcqs = []
            for mcq in mcqs[:num_questions]:
                validated_mcq = self._validate_mcq(mcq, num_choices)
                if validated_mcq:
                    validated_mcqs.append(validated_mcq)
            
            return validated_mcqs
            
        except Exception as e:
            raise Exception(f"Error generating MCQs: {str(e)}")
    
    def _create_prompt(self, topic: str, num_questions: int, difficulty: str, num_choices: int) -> str:
        """Create the prompt for MCQ generation"""
        letters = ['A', 'B', 'C', 'D', 'E'][:num_choices]
        letters_str = ', '.join(letters)
        
        return f"""Generate exactly {num_questions} multiple-choice questions on the topic: "{topic}".

CRITICAL REQUIREMENTS:
1. Difficulty level: {difficulty}
2. Each question must have exactly {num_choices} options ({letters_str})
3. Include exactly one correct answer per question
4. Provide a clear, concise, and FACTUALLY ACCURATE explanation for each answer
5. Output format must be a valid JSON array, nothing else
6. Questions must be EDUCATIONALLY SOUND and FACTUALLY CORRECT

JSON FORMAT (STRICT):
[
{{
    "question": "Question text here?",
    "options": {{
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
    }},
    "correct_answer": "A",
    "explanation": "Brief explanation why this is correct"
}}
]

IMPORTANT GUIDELINES:
- Questions should test understanding, not just memorization
- Make incorrect options plausible but clearly wrong
- Cover different aspects of the topic
- Ensure educational value and factual accuracy
- Return ONLY the JSON array
"""
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to Gemini using the new SDK"""
        try:
            # Create a system message and user message combined
            full_prompt = f"""You are an expert educational content creator. You ONLY output valid JSON arrays. Never add any text outside the JSON structure.

{prompt}"""
            
            # Generate response using the new SDK
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=self.model.temperature,
                    max_output_tokens=self.model.max_tokens,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ]
                )
            )
            
            # Extract text from response
            if response.text:
                return response.text.strip()
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            if "API key" in str(e).lower():
                raise Exception("Invalid Google API key. Please check your GOOGLE_API_KEY.")
            elif "quota" in str(e).lower() or "rate limit" in str(e).lower():
                raise Exception("Rate limit exceeded. Please try again later.")
            elif "safety" in str(e).lower():
                # Handle safety-related errors by retrying with more permissive settings
                raise Exception("Content blocked by safety filters. Try a different topic.")
            else:
                raise Exception(f"Gemini API request failed: {str(e)}")
    
    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse the API response to extract MCQs"""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Try to find JSON in the response
            start_idx = cleaned_text.find('[')
            end_idx = cleaned_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = cleaned_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    return parsed
            
            # Try direct JSON parsing
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                for key in ['questions', 'mcqs', 'data']:
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
            
        except json.JSONDecodeError:
            pass
        
        # Fallback to text extraction
        return self._extract_mcqs_from_text(response_text)
    
    def _extract_mcqs_from_text(self, text: str) -> List[Dict]:
        """Extract MCQs from text when JSON parsing fails"""
        mcqs = []
        lines = text.split('\n')
        
        current_mcq = {}
        current_options = {}
        collecting_options = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            if line.endswith('?') and len(line) > 10:
                if current_mcq and current_options:
                    mcqs.append(current_mcq)
                
                current_mcq = {
                    "question": line,
                    "options": {},
                    "correct_answer": "",
                    "explanation": ""
                }
                current_options = {}
                collecting_options = True
            
            elif collecting_options and len(line) > 2:
                first_char = line[0].upper()
                if first_char in ['A', 'B', 'C', 'D', 'E'] and line[1] in ['.', ':', ')']:
                    option_key = first_char
                    option_text = line[2:].strip()
                    current_options[option_key] = option_text
            
            elif 'correct' in line.lower() or 'answer:' in line.lower():
                if current_mcq and current_options:
                    current_mcq["options"] = current_options
                    for char in line.upper():
                        if char in ['A', 'B', 'C', 'D', 'E']:
                            current_mcq["correct_answer"] = char
                            break
                    collecting_options = False
            
            elif 'explanation' in line.lower():
                if current_mcq:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        current_mcq["explanation"] = parts[1].strip()
        
        if current_mcq and current_options:
            current_mcq["options"] = current_options
            if current_mcq.get("question") and not current_mcq.get("correct_answer"):
                current_mcq["correct_answer"] = list(current_options.keys())[0]
            mcqs.append(current_mcq)
        
        return mcqs
    
    def _validate_mcq(self, mcq: Dict, num_choices: int) -> Optional[Dict]:
        """Validate and clean a single MCQ"""
        if not isinstance(mcq, dict):
            return None
        
        if 'question' not in mcq or not mcq['question']:
            return None
        
        mcq['question'] = mcq['question'].strip()
        if len(mcq['question']) < 5:
            return None
        
        if 'options' not in mcq or not isinstance(mcq['options'], dict):
            return None
        
        cleaned_options = {}
        for key, value in mcq['options'].items():
            if isinstance(key, str) and key.upper() in ['A', 'B', 'C', 'D', 'E']:
                clean_key = key.upper()
                if isinstance(value, str) and value.strip():
                    cleaned_options[clean_key] = value.strip()
        
        if len(cleaned_options) < 2:
            return None
        
        mcq['options'] = cleaned_options
        
        if 'correct_answer' not in mcq or not mcq['correct_answer']:
            mcq['correct_answer'] = list(cleaned_options.keys())[0]
        
        if mcq['correct_answer'] not in cleaned_options:
            mcq['correct_answer'] = list(cleaned_options.keys())[0]
        
        if 'explanation' not in mcq or not mcq['explanation']:
            mcq['explanation'] = "No explanation provided."
        
        return mcq


def create_env_template():
    """Create .env file template"""
    env_content = """# Google Gemini API Configuration
# Get your API key from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here

# Available Models:
# - gemini-2.5-flash (Fast, efficient - recommended)
# - gemini-2.5-flash-lite (Lightning fast)
# - gemini-2.0-flash (Balanced performance)
# - gemini-2.0-flash-lite (Very fast)
# - gemma-3-27b-it (Open model, good for technical)
# - gemma-3-12b-it (Smaller open model)
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    return env_content


# Simple test function
def test_gemini_client():
    """Test the Gemini client with a simple query"""
    try:
        client = GeminiClient()
        print("✅ GeminiClient initialized successfully")
        
        # Test model info
        info = client.get_model_info()
        print(f"   Current model: {info['name']}")
        
        # Test generating a single question
        mcqs = client.generate_mcqs(
            topic="Python programming",
            num_questions=1,
            difficulty="easy",
            num_choices=4
        )
        
        if mcqs:
            print(f"✅ Generated {len(mcqs)} question")
            print(f"   Question: {mcqs[0]['question'][:50]}...")
            return True
        else:
            print("❌ No questions generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Gemini Client...")
    test_gemini_client()