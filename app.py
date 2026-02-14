# app.py
import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Import Gemini client from llm_config
from llm_config import GeminiClient, create_env_template

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    
    initialize_session_state()
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📝 Quiz: {topic}")
    with col2:
        st.metric("Total Questions", len(mcqs))
    with col3:
        difficulty = st.session_state.get('difficulty', 'Medium')
        st.metric("Difficulty", difficulty.title() if isinstance(difficulty, str) else "Medium")
    
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
        if st.button("📤 Submit Quiz", use_container_width=True, type="secondary", disabled=submit_disabled):
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
    
    start_idx = (st.session_state.current_page - 1) * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(mcqs))
    page_mcqs = mcqs[start_idx:end_idx]
    
    for i, mcq in enumerate(page_mcqs, start=start_idx + 1):
        with st.container():
            st.markdown(f"### Question {i}")
            st.markdown(f"**{mcq['question']}**")
            
            options = mcq['options']
            option_keys = list(options.keys())
            question_key = f"quiz_q_{i}"
            
            if not st.session_state.show_answers:
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
                
                if question_key in st.session_state.user_answers:
                    st.success("✓ Answer saved")
                
            else:
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
                
                with st.expander("📖 Explanation"):
                    st.info(mcq['explanation'])
            
            st.markdown("---")
    
    answered = len(st.session_state.user_answers)
    total = len(mcqs)
    progress = answered / total if total > 0 else 0
    
    st.markdown(f"**Progress:** {answered}/{total} questions answered ({progress:.0%})")
    st.progress(progress)
    
    if not st.session_state.quiz_submitted:
        st.markdown("---")
        submit_col1, submit_col2 = st.columns([3, 1])
        with submit_col2:
            submit_disabled = answered < total
            if st.button("🚀 Submit Final Answers", use_container_width=True, type="primary", disabled=submit_disabled):
                st.session_state.quiz_submitted = True
                st.session_state.show_answers = True
                st.rerun()
    
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Your Score", f"{score_percentage:.1f}%")
    with col2:
        st.metric("✅ Correct", f"{correct_count}/{total_questions}")
    with col3:
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0
        st.metric("📈 Accuracy", f"{accuracy:.1f}%")
    
    st.progress(score_percentage / 100)
    
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
    initialize_session_state()
    
    total_pages = (len(mcqs) + questions_per_page - 1) // questions_per_page
    start_idx = (st.session_state.summary_page - 1) * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(mcqs))
    page_mcqs = mcqs[start_idx:end_idx]
    
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
    initialize_session_state()
    
    total_pages = (len(mcqs) + questions_per_page - 1) // questions_per_page
    start_idx = (st.session_state.list_page - 1) * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(mcqs))
    page_mcqs = mcqs[start_idx:end_idx]
    
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
        .footer {
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid #E5E7EB;
            color: #6B7280;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎓 AI MCQ Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate interactive quizzes on any topic using Google Gemini AI</p>', unsafe_allow_html=True)
    
    # Initialize Gemini client
    try:
        # Check if .env file exists, create template if not
        if not os.path.exists(".env"):
            with st.expander("📁 Create .env file"):
                st.code(create_env_template(), language="bash")
                if st.button("Create .env file now"):
                    create_env_template()
                    st.success("✅ .env file created! Please edit it with your actual API key and restart the app.")
        
        client = GeminiClient()  # Changed from GroqClient to GeminiClient
        api_key_available = True
    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
        api_key_available = False
    
    if not api_key_available:
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Model selection
        st.markdown("#### 🤖 Select Gemini Model")
        available_models = client.get_available_models()
        model_options = {m["id"]: f"{m['name']} - {m['description']}" for m in available_models}
        
        selected_model = st.selectbox(
            "Choose Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        # Show current model info
        model_info = client.get_model_info()
        st.info(f"**Current:** {model_info['name']}\n\n{model_info['description']}")
        
        st.markdown("---")
        
        # Generation settings
        st.markdown("#### 🎯 Quiz Settings")
        
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=50,
            value=10
        )
        
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["easy", "medium", "hard"],
            value="medium",
            format_func=lambda x: f"{x.title()} 🎯"
        )
        
        num_choices = st.selectbox(
            "Choices per Question",
            options=[3, 4, 5],
            index=1,
            format_func=lambda x: f"{x} options"
        )
        
        questions_per_page = st.selectbox(
            "Questions per Page",
            options=[3, 5, 10, 20],
            index=1
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
        st.info("💡 **Tip:** Be specific with your topic for better questions!")
        
        # Add model info note
        st.markdown("---")
        st.caption("Powered by Google Gemini AI • Free tier available")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if selected_example == "Custom topic...":
            topic = st.text_input(
                "🎯 Enter Your Topic",
                placeholder="e.g., Machine Learning algorithms, Renaissance art, Quantum physics basics..."
            )
        else:
            topic = selected_example
            st.markdown(f"**Selected Topic:** {topic}")
            topic = st.text_input(
                "Or enter a custom topic:",
                value=topic
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
        # Update model if changed
        client.set_model(selected_model)
        
        with st.spinner(f"Creating {num_questions} questions on '{topic}' using {model_info['name']}..."):
            try:
                st.session_state.difficulty = difficulty
                
                mcqs = client.generate_mcqs(
                    topic=topic,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    num_choices=num_choices
                )
                
                if mcqs:
                    st.session_state.generated_mcqs = mcqs
                    st.session_state.current_topic = topic
                    st.session_state.questions_per_page = questions_per_page
                    
                    st.success(f"✅ Generated {len(mcqs)} questions successfully using {model_info['name']}!")
                else:
                    st.error("❌ No questions were generated. Please try a different topic.")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display results
    if 'generated_mcqs' in st.session_state and st.session_state.generated_mcqs:
        mcqs = st.session_state.generated_mcqs
        topic = st.session_state.get('current_topic', 'Generated Topic')
        questions_per_page = st.session_state.get('questions_per_page', 5)
        
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
                json_data = json.dumps(mcqs, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"quiz_{topic[:30].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
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
                    use_container_width=True
                )
            
            with col3:
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
                    use_container_width=True
                )
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong>🎓 AI MCQ Generator</strong><br>
            Powered by Google Gemini AI • Built with Streamlit
        </div>
        <div style="text-align: right;">
            <small>v3.0 • {datetime.now().strftime("%Y-%m-%d")}</small><br>
            <small>Free tier: 60 requests per minute</small>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()