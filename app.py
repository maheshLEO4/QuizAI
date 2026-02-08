import streamlit as st
import os
import json
import requests
from typing import List, Dict, Optional
import pandas as pd
import base64
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GroqMCQGenerator:
    def __init__(self, api_key: str = None):
        """
        Initialize the Groq MCQ Generator
        """
        # Try to get API key from parameter, then environment variable
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set.\n\n"
                "Please create a `.env` file in the same directory with this content:\n"
                "GROQ_API_KEY=your_actual_api_key_here\n\n"
                "Get your API key from: https://console.groq.com/keys"
            )
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Available free models on Groq
        self.available_models = {
            "llama-3.1-8b-instant": {"name": "Llama 3.1 8B Instant", "description": "Fast and efficient"},
            "llama-3.2-1b-preview": {"name": "Llama 3.2 1B Preview", "description": "Lightweight model"},
            "llama-3.2-3b-preview": {"name": "Llama 3.2 3B Preview", "description": "Balanced performance"},
            "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "description": "High quality, larger context"},
            "gemma2-9b-it": {"name": "Gemma2 9B", "description": "Google's efficient model"}
        }
    
    def generate_mcqs(self, 
                     topic: str, 
                     num_questions: int = 5, 
                     difficulty: str = "medium",
                     model: str = "llama-3.1-8b-instant",
                     num_choices: int = 4) -> List[Dict]:
        """
        Generate MCQs based on a given topic
        """
        if model not in self.available_models:
            model = "llama-3.1-8b-instant"
        
        prompt = self._create_prompt(topic, num_questions, difficulty, num_choices)
        
        try:
            model_name = self.available_models[model]["name"]
            with st.spinner(f"🤖 Generating {num_questions} MCQs on '{topic}' using {model_name}..."):
                response = self._call_groq_api(prompt, model)
                mcqs = self._parse_response(response)
                
                # Validate and clean the MCQs
                validated_mcqs = []
                for mcq in mcqs[:num_questions]:
                    validated_mcq = self._validate_mcq(mcq, num_choices)
                    if validated_mcq:
                        validated_mcqs.append(validated_mcq)
                
                return validated_mcqs
            
        except Exception as e:
            st.error(f"❌ Error generating MCQs: {str(e)}")
            return []
    
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
    7. If generating questions on technical topics, ensure definitions are precise and accurate

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
        "explanation": "Brief explanation why this is correct, ensuring factual accuracy"
    }}
    ]

    IMPORTANT GUIDELINES:
    - Questions should test understanding, not just memorization
    - Make incorrect options plausible but clearly wrong
    - Cover different aspects of the topic
    - Ensure educational value and factual accuracy
    - For technical topics: verify definitions against authoritative sources
    - Return ONLY the JSON array
    - Double-check that all information is correct before including
    """
    
    def _call_groq_api(self, prompt: str, model: str) -> str:
        """Make API call to Groq"""
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert educational content creator. You ONLY output valid JSON arrays. Never add any text outside the JSON structure."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            if "401" in str(e):
                raise Exception("Invalid API key. Please check your GROQ_API_KEY environment variable.")
            elif "429" in str(e):
                raise Exception("Rate limit exceeded. Please try again later.")
            else:
                raise Exception(f"API request failed: {str(e)}")
    
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
            
            # Skip empty lines
            if not line:
                continue
            
            # Detect question (ends with ? or starts with Q/Question/number)
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
            
            # Detect options
            elif collecting_options and len(line) > 2:
                first_char = line[0].upper()
                if first_char in ['A', 'B', 'C', 'D', 'E'] and line[1] in ['.', ':', ')']:
                    option_key = first_char
                    option_text = line[2:].strip()
                    current_options[option_key] = option_text
            
            # Detect correct answer
            elif 'correct' in line.lower() or 'answer:' in line.lower():
                if current_mcq and current_options:
                    current_mcq["options"] = current_options
                    for char in line.upper():
                        if char in ['A', 'B', 'C', 'D', 'E']:
                            current_mcq["correct_answer"] = char
                            break
                    collecting_options = False
            
            # Detect explanation
            elif 'explanation' in line.lower():
                if current_mcq:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        current_mcq["explanation"] = parts[1].strip()
        
        # Add the last MCQ
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
        
        # Check required fields
        if 'question' not in mcq or not mcq['question']:
            return None
        
        # Clean question
        mcq['question'] = mcq['question'].strip()
        if len(mcq['question']) < 5:
            return None
        
        # Ensure options exist and are valid
        if 'options' not in mcq or not isinstance(mcq['options'], dict):
            return None
        
        # Clean options
        cleaned_options = {}
        for key, value in mcq['options'].items():
            if isinstance(key, str) and key.upper() in ['A', 'B', 'C', 'D', 'E']:
                clean_key = key.upper()
                if isinstance(value, str) and value.strip():
                    cleaned_options[clean_key] = value.strip()
        
        if len(cleaned_options) < 2:
            return None
        
        mcq['options'] = cleaned_options
        
        # Set correct answer if missing
        if 'correct_answer' not in mcq or not mcq['correct_answer']:
            mcq['correct_answer'] = list(cleaned_options.keys())[0]
        
        # Ensure correct answer is valid
        if mcq['correct_answer'] not in cleaned_options:
            mcq['correct_answer'] = list(cleaned_options.keys())[0]
        
        # Add explanation if missing
        if 'explanation' not in mcq or not mcq['explanation']:
            mcq['explanation'] = "No explanation provided."
        
        return mcq


def initialize_session_state():
    """Initialize session state variables"""
    if 'quiz_initialized' not in st.session_state:
        st.session_state.quiz_initialized = True
        st.session_state.user_answers = {}
        st.session_state.show_answers = False
        st.session_state.quiz_submitted = False
        st.session_state.current_page = 1
        st.session_state.summary_page = 1
        st.session_state.list_page = 1


def display_mcqs_with_quiz(mcqs: List[Dict], topic: str, questions_per_page: int = 5):
    """Display MCQs as an interactive quiz with pagination"""
    if not mcqs:
        st.warning("No MCQs generated. Please try again with different parameters.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    st.markdown("---")
    
    # Header with topic info
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📝 Quiz: {topic}")
    with col2:
        st.metric("Total Questions", len(mcqs))
    with col3:
        difficulty = st.session_state.get('difficulty', 'Medium')
        if isinstance(difficulty, str):
            st.metric("Difficulty", difficulty.title())
        else:
            st.metric("Difficulty", "Medium")
    
    # Quiz controls
    st.markdown("---")
    
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    
    with control_col1:
        if st.button("🔄 Start New Quiz", use_container_width=True, type="primary"):
            st.session_state.user_answers = {}
            st.session_state.show_answers = False
            st.session_state.quiz_submitted = False
            st.session_state.current_page = 1
            st.rerun()
    
    with control_col2:
        submit_disabled = st.session_state.quiz_submitted or len(st.session_state.user_answers) < len(mcqs)
        if st.button("📤 Submit Quiz", 
                    use_container_width=True, 
                    type="secondary",
                    disabled=submit_disabled):
            st.session_state.quiz_submitted = True
            st.session_state.show_answers = True
            st.rerun()
    
    with control_col3:
        if st.button("👁️ Show Answers", use_container_width=True):
            st.session_state.show_answers = True
            st.rerun()
    
    with control_col4:
        if st.button("🙈 Hide Answers", use_container_width=True):
            st.session_state.show_answers = False
            st.rerun()
    
    st.markdown("---")
    
    # Pagination
    total_pages = (len(mcqs) + questions_per_page - 1) // questions_per_page
    if total_pages > 1:
        page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
        with page_col1:
            if st.button("◀ Previous", disabled=st.session_state.current_page == 1):
                st.session_state.current_page -= 1
                st.rerun()
        with page_col2:
            st.markdown(f"**Page {st.session_state.current_page} of {total_pages}**")
        with page_col3:
            if st.button("Next ▶", disabled=st.session_state.current_page == total_pages):
                st.session_state.current_page += 1
                st.rerun()
        st.markdown("---")
    
    # Calculate question range for current page
    start_idx = (st.session_state.current_page - 1) * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(mcqs))
    page_mcqs = mcqs[start_idx:end_idx]
    
    # Display questions for current page
    for i, mcq in enumerate(page_mcqs, start=start_idx + 1):
        with st.container():
            st.markdown(f"### Question {i}")
            st.markdown(f"**{mcq['question']}**")
            
            # Get options
            options = mcq['options']
            option_keys = list(options.keys())
            
            # Create a unique key for each question
            question_key = f"quiz_q_{i}"
            
            if not st.session_state.show_answers:
                # Quiz mode - user selects answers
                user_answer = st.radio(
                    f"Select your answer for Question {i}:",
                    options=option_keys,
                    format_func=lambda x: f"**{x}.** {options[x]}",
                    key=question_key,
                    index=None,
                    label_visibility="visible"
                )
                
                if user_answer:
                    st.session_state.user_answers[question_key] = user_answer
                
                # Show answer status
                if question_key in st.session_state.user_answers:
                    st.success("✓ Answer saved")
                
            else:
                # Show answers mode
                user_answer = st.session_state.user_answers.get(question_key, "")
                correct_answer = mcq['correct_answer']
                
                st.markdown("**Options:**")
                for option_key in option_keys:
                    option_text = options[option_key]
                    is_correct = option_key == correct_answer
                    is_user_answer = option_key == user_answer
                    
                    if is_correct:
                        st.markdown(f"✅ **{option_key}. {option_text}** *(Correct Answer)*")
                    elif is_user_answer and not is_correct:
                        st.markdown(f"❌ **{option_key}. {option_text}** *(Your Answer)*")
                    else:
                        st.markdown(f"{option_key}. {option_text}")
                
                # Show explanation
                with st.expander("📖 Explanation"):
                    st.info(mcq['explanation'])
            
            st.markdown("---")
    
    # Progress indicator
    answered = len(st.session_state.user_answers)
    total = len(mcqs)
    progress = answered / total if total > 0 else 0
    
    st.markdown(f"**Progress:** {answered}/{total} questions answered ({progress:.0%})")
    st.progress(progress)
    
    # Submit button at the bottom
    if not st.session_state.quiz_submitted:
        st.markdown("---")
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col2:
            submit_disabled = answered < total
            if st.button("🚀 Submit Final Answers", 
                        use_container_width=True, 
                        type="primary",
                        disabled=submit_disabled):
                st.session_state.quiz_submitted = True
                st.session_state.show_answers = True
                st.rerun()
    
    # Display quiz results if submitted
    if st.session_state.quiz_submitted and st.session_state.show_answers:
        calculate_and_display_score(mcqs)


def calculate_and_display_score(mcqs: List[Dict]):
    """Calculate and display the quiz score"""
    correct_count = 0
    total_questions = len(mcqs)
    
    for i in range(1, total_questions + 1):
        question_key = f"quiz_q_{i}"
        if question_key in st.session_state.user_answers:
            user_answer = st.session_state.user_answers[question_key]
            if user_answer == mcqs[i-1]['correct_answer']:
                correct_count += 1
    
    score_percentage = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    st.markdown("---")
    st.markdown("## 🏆 Quiz Results")
    
    # Create metrics with emojis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Your Score", f"{score_percentage:.1f}%")
    
    with col2:
        st.metric("✅ Correct", f"{correct_count}/{total_questions}")
    
    with col3:
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
        st.metric("📈 Accuracy", f"{accuracy:.1f}%")
    
    # Progress bar with color coding
    st.progress(score_percentage / 100)
    
    # Performance message with emojis
    if score_percentage >= 90:
        st.success("🏅 **Outstanding!** Perfect or near-perfect score!")
    elif score_percentage >= 80:
        st.success("🎉 **Excellent!** You've mastered this topic!")
    elif score_percentage >= 70:
        st.info("👍 **Very good!** Strong understanding of the material.")
    elif score_percentage >= 60:
        st.info("📚 **Good job!** You understand the basics well.")
    elif score_percentage >= 50:
        st.warning("🤔 **Fair.** Review the explanations to improve.")
    else:
        st.error("📖 **Needs improvement.** Study the material and try again.")
    
    # Detailed breakdown
    with st.expander("📊 View Detailed Breakdown"):
        for i in range(1, total_questions + 1):
            question_key = f"quiz_q_{i}"
            user_answer = st.session_state.user_answers.get(question_key, "Not answered")
            correct_answer = mcqs[i-1]['correct_answer']
            is_correct = user_answer == correct_answer
            
            status = "✅" if is_correct else "❌"
            st.markdown(f"{status} **Q{i}:** Your answer: {user_answer} | Correct: {correct_answer}")


def display_mcqs_summary(mcqs: List[Dict], questions_per_page: int = 10):
    """Display MCQs in a summary format with pagination"""
    if not mcqs:
        return
    
    st.markdown("### 📋 Questions Summary")
    
    # Initialize pagination for summary view
    initialize_session_state()
    
    total_pages = (len(mcqs) + questions_per_page - 1) // questions_per_page
    start_idx = (st.session_state.summary_page - 1) * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(mcqs))
    page_mcqs = mcqs[start_idx:end_idx]
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀ Prev", key="prev_summary", disabled=st.session_state.summary_page == 1):
                st.session_state.summary_page -= 1
                st.rerun()
        with col2:
            st.markdown(f"**Page {st.session_state.summary_page} of {total_pages}**")
        with col3:
            if st.button("Next ▶", key="next_summary", disabled=st.session_state.summary_page == total_pages):
                st.session_state.summary_page += 1
                st.rerun()
    
    # Display questions
    for i, mcq in enumerate(page_mcqs, start=start_idx + 1):
        with st.expander(f"Question {i}: {mcq['question'][:80]}..."):
            st.markdown(f"**Question:** {mcq['question']}")
            st.markdown("**Options:**")
            
            for option, text in mcq['options'].items():
                if option == mcq['correct_answer']:
                    st.markdown(f"✅ **{option}. {text}** *(Correct Answer)*")
                else:
                    st.markdown(f"{option}. {text}")
            
            st.markdown(f"**Explanation:** {mcq['explanation']}")


def display_mcqs_list(mcqs: List[Dict], questions_per_page: int = 5):
    """Display MCQs in a list format with pagination"""
    if not mcqs:
        return
    
    st.markdown("### 📝 Questions List")
    
    # Initialize pagination for list view
    initialize_session_state()
    
    total_pages = (len(mcqs) + questions_per_page - 1) // questions_per_page
    start_idx = (st.session_state.list_page - 1) * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(mcqs))
    page_mcqs = mcqs[start_idx:end_idx]
    
    # Pagination controls
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("◀ Prev", key="prev_list", disabled=st.session_state.list_page == 1):
                st.session_state.list_page -= 1
                st.rerun()
        with col2:
            st.markdown(f"**Page {st.session_state.list_page} of {total_pages}**")
        with col3:
            if st.button("Next ▶", key="next_list", disabled=st.session_state.list_page == total_pages):
                st.session_state.list_page += 1
                st.rerun()
    
    # Display questions
    for i, mcq in enumerate(page_mcqs, start=start_idx + 1):
        with st.container():
            st.markdown(f"---")
            st.markdown(f"#### Q{i}: {mcq['question']}")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                for option, text in mcq['options'].items():
                    st.markdown(f"**{option}.** {text}")
            
            with col2:
                st.success(f"**Answer: {mcq['correct_answer']}**")
            
            with st.expander("View Explanation"):
                st.info(mcq['explanation'])


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="AI MCQ Generator",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 0.5rem;
            font-weight: 800;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #4B5563;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stButton button {
            transition: all 0.3s ease;
            font-weight: 600;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .card {
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .success-card {
            background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
            border-left: 6px solid #10B981;
        }
        .info-card {
            background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%);
            border-left: 6px solid #3B82F6;
        }
        .model-card {
            background-color: #F8FAFC;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
        }
        .model-card:hover {
            border-color: #4F46E5;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.1);
        }
        .selected-model {
            border-color: #4F46E5;
            background-color: #EEF2FF;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #E5E7EB;
            color: #6B7280;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎓 AI MCQ Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate interactive quizzes on any topic using AI</p>', unsafe_allow_html=True)
    
    # Check for API key and initialize generator
    try:
        generator = GroqMCQGenerator()
        api_key_available = True
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        
        # Create .env file template
        env_content = """GROQ_API_KEY=your_api_key_here
        
# Get your API key from:
# 1. Go to https://console.groq.com
# 2. Sign up for free account
# 3. Create API key from dashboard
# 4. Replace 'your_api_key_here' with your actual key
"""
        
        with st.expander("📁 How to create .env file"):
            st.code(env_content, language="bash")
            
            # Create .env file button
            if st.button("Create .env file template"):
                try:
                    with open(".env", "w") as f:
                        f.write(env_content)
                    st.success("✅ .env file created! Now edit it with your actual API key.")
                except Exception as e:
                    st.error(f"Failed to create .env file: {e}")
        
        api_key_available = False
    
    if not api_key_available:
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection with cards
        st.markdown("#### 🤖 Select AI Model")
        selected_model = st.selectbox(
            "Choose Model",
            options=list(generator.available_models.keys()),
            format_func=lambda x: f"{generator.available_models[x]['name']} - {generator.available_models[x]['description']}",
            index=0
        )
        
        # Display model info
        model_info = generator.available_models[selected_model]
        st.markdown(f"""
        <div class="model-card selected-model">
        <strong>📊 Selected Model:</strong><br>
        {model_info['name']}<br>
        <small>{model_info['description']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Generation settings
        st.markdown("#### 🎯 Quiz Settings")
        
        # Number of questions with limits
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum 50 questions per generation"
        )
        
        # Difficulty level
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["easy", "medium", "hard"],
            value="medium",
            format_func=lambda x: f"{x.title()} 🎯"
        )
        
        # Number of choices
        num_choices = st.selectbox(
            "Choices per Question",
            options=[3, 4, 5],
            index=1,
            format_func=lambda x: f"{x} options"
        )
        
        # Questions per page for quiz
        questions_per_page = st.selectbox(
            "Questions per Page",
            options=[3, 5, 10, 20],
            index=1,
            help="For better navigation in quiz mode"
        )
        
        st.markdown("---")
        
        # Example topics
        st.markdown("#### 📚 Quick Topics")
        example_topics = [
            "Python programming basics",
            "World History - Ancient Civilizations",
            "Biology - Cell Structure",
            "Mathematics - Algebra Fundamentals",
            "Physics - Newton's Laws",
            "Chemistry - Periodic Table",
            "Computer Science - Data Structures",
            "Economics - Supply and Demand"
        ]
        
        selected_example = st.selectbox(
            "Try an example topic:",
            ["Custom topic..."] + example_topics
        )
        
        st.markdown("---")
        
        # Tips
        st.markdown("#### 💡 Tips")
        st.info("""
        • Be specific with your topic
        • Start with medium difficulty
        • 4 choices is optimal for quizzes
        • Use pagination for large quizzes
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Topic input
        if selected_example == "Custom topic...":
            topic = st.text_input(
                "🎯 Enter Your Topic",
                placeholder="e.g., Machine Learning algorithms, Renaissance art, Quantum physics basics...",
                help="The more specific, the better the questions!"
            )
        else:
            topic = selected_example
            st.markdown(f"""
            <div class="info-card">
            <strong>Selected Topic:</strong> {topic}<br>
            <small>You can still type a custom topic below</small>
            </div>
            """, unsafe_allow_html=True)
            topic = st.text_input(
                "Or enter a custom topic:",
                value=topic,
                placeholder="Type your custom topic here..."
            )
    
    with col2:
        st.markdown("###")
        generate_btn = st.button(
            "✨ Generate Quiz",
            type="primary",
            use_container_width=True,
            disabled=not topic
        )
    
    # Generate MCQs
    if generate_btn and topic:
        with st.spinner(f"Creating {num_questions} questions on '{topic}'..."):
            try:
                # Store settings in session state
                st.session_state.difficulty = difficulty
                
                mcqs = generator.generate_mcqs(
                    topic=topic,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    model=selected_model,
                    num_choices=num_choices
                )
                
                if mcqs:
                    st.session_state.generated_mcqs = mcqs
                    st.session_state.current_topic = topic
                    st.session_state.questions_per_page = questions_per_page
                    
                    # Show success message
                    st.markdown(f"""
                    <div class="success-card">
                    <h3>✅ Quiz Generated Successfully!</h3>
                    <p><strong>Topic:</strong> {topic}</p>
                    <p><strong>Questions:</strong> {len(mcqs)} generated</p>
                    <p><strong>Model:</strong> {model_info['name']}</p>
                    <p><strong>Difficulty:</strong> {difficulty.title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ No questions were generated. Please try a different topic or adjust settings.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display results
    if 'generated_mcqs' in st.session_state and st.session_state.generated_mcqs:
        mcqs = st.session_state.generated_mcqs
        topic = st.session_state.get('current_topic', 'Generated Topic')
        questions_per_page = st.session_state.get('questions_per_page', 5)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Interactive Quiz", "📋 Questions List", "📊 Summary", "💾 Export"])
        
        with tab1:
            display_mcqs_with_quiz(mcqs, topic, questions_per_page)
        
        with tab2:
            display_mcqs_list(mcqs, questions_per_page)
        
        with tab3:
            display_mcqs_summary(mcqs, questions_per_page)
        
        with tab4:
            st.markdown("### 💾 Export Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON Export
                json_data = json.dumps(mcqs, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"quiz_{topic[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    help="JSON format for developers and APIs"
                )
            
            with col2:
                # CSV Export
                csv_data = []
                for i, mcq in enumerate(mcqs, 1):
                    row = {
                        "Question_Number": i,
                        "Question": mcq['question'],
                        "Correct_Answer": mcq['correct_answer'],
                        "Explanation": mcq['explanation']
                    }
                    for opt_key, opt_val in mcq['options'].items():
                        row[f"Option_{opt_key}"] = opt_val
                    csv_data.append(row)
                
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_string,
                    file_name=f"quiz_{topic[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="CSV format for spreadsheets and data analysis"
                )
            
            with col3:
                # Text Export
                text_output = f"QUIZ: {topic}\n"
                text_output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                text_output += f"Total Questions: {len(mcqs)}\n"
                text_output += "=" * 60 + "\n\n"
                
                for i, mcq in enumerate(mcqs, 1):
                    text_output += f"QUESTION {i}:\n"
                    text_output += f"{mcq['question']}\n\n"
                    text_output += "OPTIONS:\n"
                    for opt_key, opt_val in mcq['options'].items():
                        text_output += f"  {opt_key}. {opt_val}\n"
                    text_output += f"\n✓ CORRECT ANSWER: {mcq['correct_answer']}\n"
                    text_output += f"📖 EXPLANATION: {mcq['explanation']}\n\n"
                    text_output += "-" * 40 + "\n\n"
                
                st.download_button(
                    label="📝 Download Text",
                    data=text_output,
                    file_name=f"quiz_{topic[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    help="Plain text format for printing and sharing"
                )
            
            # Preview
            with st.expander("🔍 Preview Export"):
                if len(mcqs) > 0:
                    st.json(mcqs[0])
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong>🎓 AI MCQ Generator</strong><br>
            Powered by Groq AI • Built with Streamlit
        </div>
        <div style="text-align: right;">
            <small>v2.0 • {datetime.now().strftime("%Y-%m-%d")}</small><br>
            <small>Questions per page limit: 50</small>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()