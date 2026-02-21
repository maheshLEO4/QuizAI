# app.py
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict

from llm_config import LLMClient, create_env_template

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuizAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background: #0a0a0f; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e30;
}
[data-testid="stSidebar"] * { color: #c8c8d8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #888 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: .06em; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #e8e8ff 0%, #a78bfa 50%, #60a5fa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
    letter-spacing: -.02em;
}
.hero p {
    color: #6b6b85;
    font-size: 1rem;
    margin-top: .6rem;
    font-weight: 300;
}

/* Cards */
.card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Question card */
.q-card {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    transition: border-color .2s;
}
.q-card:hover { border-color: #3d3d60; }
.q-num {
    font-size: .72rem;
    font-weight: 600;
    color: #6060a0;
    text-transform: uppercase;
    letter-spacing: .12em;
    margin-bottom: .5rem;
}
.q-text {
    color: #e2e2f2;
    font-size: 1.05rem;
    font-weight: 500;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: .88rem !important;
    transition: all .2s !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    box-shadow: 0 0 20px rgba(124,58,237,.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 30px rgba(124,58,237,.5) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="secondary"] {
    background: #1e1e30 !important;
    color: #c8c8d8 !important;
    border: 1px solid #2a2a45 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #252540 !important;
    color: white !important;
}

/* Radio overrides */
.stRadio > div { gap: .4rem; }
.stRadio > div > label {
    background: #1a1a28 !important;
    border: 1px solid #252538 !important;
    border-radius: 10px !important;
    padding: .7rem 1rem !important;
    color: #c0c0d8 !important;
    transition: all .15s !important;
    cursor: pointer !important;
}
.stRadio > div > label:hover {
    border-color: #5a5af0 !important;
    color: white !important;
}

/* Input */
.stTextInput > div > div > input {
    background: #13131f !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 12px !important;
    color: #e2e2f2 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: .7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,.2) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f18;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e1e30;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: #6b6b85 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
    padding: .45rem 1.1rem !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: #1e1e30 !important;
    color: #e2e2f2 !important;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa) !important;
    border-radius: 99px !important;
}
.stProgress > div > div {
    background: #1e1e30 !important;
    border-radius: 99px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #13131f;
    border: 1px solid #1e1e30;
    border-radius: 12px;
    padding: .8rem 1rem;
}
[data-testid="stMetricValue"] { color: #e2e2f2 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #6b6b85 !important; font-size: .75rem !important; }

/* Alerts */
.stSuccess, .stInfo, .stWarning, .stError {
    border-radius: 10px !important;
    border: none !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #13131f !important;
    border-radius: 10px !important;
    color: #a0a0c0 !important;
    font-size: .88rem !important;
}
.streamlit-expanderContent {
    background: #0f0f18 !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 0 0 10px 10px !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: #1e1e30 !important;
    color: #c8c8d8 !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 10px !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    border-color: #7c3aed !important;
    color: white !important;
}

/* Divider */
hr { border-color: #1e1e30 !important; margin: 1.5rem 0 !important; }

/* Score badge */
.score-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Correct/Wrong markers */
.opt-correct { color: #4ade80 !important; font-weight: 600; }
.opt-wrong   { color: #f87171 !important; }
.opt-plain   { color: #9090b0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "answers": {},
        "show_ans": False,
        "submitted": False,
        "quiz_page": 1,
        "mcqs": [],
        "topic": "",
        "difficulty": "medium",
        "qpp": 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Client init (cached) ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client():
    return LLMClient()


# ── Quiz display ──────────────────────────────────────────────────────────────

def render_quiz(mcqs: List[Dict]):
    qpp   = st.session_state.qpp
    total = len(mcqs)
    pages = (total + qpp - 1) // qpp

    # Controls row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("↺  New Quiz", use_container_width=True, type="primary"):
            st.session_state.answers   = {}
            st.session_state.show_ans  = False
            st.session_state.submitted = False
            st.session_state.quiz_page = 1
            st.rerun()
    with c2:
        answered = len(st.session_state.answers)
        can_submit = answered >= total and not st.session_state.submitted
        if st.button("✓  Submit", use_container_width=True, type="secondary", disabled=not can_submit):
            st.session_state.submitted = True
            st.session_state.show_ans  = True
            st.rerun()
    with c3:
        if st.button("👁  Show Answers", use_container_width=True, type="secondary"):
            st.session_state.show_ans = True
            st.rerun()
    with c4:
        if st.button("🙈  Hide Answers", use_container_width=True, type="secondary"):
            st.session_state.show_ans = False
            st.rerun()

    st.markdown("")

    # Progress bar
    answered = len(st.session_state.answers)
    prog = answered / total if total else 0
    st.caption(f"{answered} / {total} answered")
    st.progress(prog)

    st.markdown("")

    # Pagination
    if pages > 1:
        pc1, pc2, pc3 = st.columns([1, 3, 1])
        with pc1:
            if st.button("← Prev", disabled=st.session_state.quiz_page == 1, use_container_width=True):
                st.session_state.quiz_page -= 1
                st.rerun()
        with pc2:
            st.markdown(
                f"<p style='text-align:center;color:#6b6b85;font-size:.85rem;margin:0'>"
                f"Page {st.session_state.quiz_page} of {pages}</p>",
                unsafe_allow_html=True
            )
        with pc3:
            if st.button("Next →", disabled=st.session_state.quiz_page == pages, use_container_width=True):
                st.session_state.quiz_page += 1
                st.rerun()
        st.markdown("")

    # Questions
    start = (st.session_state.quiz_page - 1) * qpp
    for i, mcq in enumerate(mcqs[start:start + qpp], start=start + 1):
        opts    = mcq["options"]
        key     = f"q_{i}"
        correct = mcq["correct_answer"]
        chosen  = st.session_state.answers.get(key, "")

        st.markdown(
            f'<div class="q-card">'
            f'<div class="q-num">Question {i}</div>'
            f'<div class="q-text">{mcq["question"]}</div>',
            unsafe_allow_html=True
        )

        if not st.session_state.show_ans:
            ans = st.radio(
                "Choose:",
                list(opts.keys()),
                format_func=lambda x: f"{x}.  {opts[x]}",
                key=key,
                index=None,
                label_visibility="collapsed",
            )
            if ans:
                st.session_state.answers[key] = ans
        else:
            for ok, ov in opts.items():
                if ok == correct:
                    st.markdown(f'<p class="opt-correct">✓ {ok}. {ov}</p>', unsafe_allow_html=True)
                elif ok == chosen:
                    st.markdown(f'<p class="opt-wrong">✗ {ok}. {ov}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="opt-plain">{ok}. {ov}</p>', unsafe_allow_html=True)
            with st.expander("Explanation"):
                st.markdown(f"<span style='color:#b0b0cc'>{mcq['explanation']}</span>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Score panel
    if st.session_state.submitted:
        render_score(mcqs)


def render_score(mcqs: List[Dict]):
    correct = sum(
        1 for i, m in enumerate(mcqs, 1)
        if st.session_state.answers.get(f"q_{i}") == m["correct_answer"]
    )
    total   = len(mcqs)
    pct     = correct / total * 100 if total else 0

    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;padding:1.5rem 0'>"
        f"<div style='color:#6b6b85;font-size:.8rem;text-transform:uppercase;letter-spacing:.1em'>Your Score</div>"
        f"<div class='score-badge'>{pct:.0f}%</div>"
        f"<div style='color:#6b6b85;margin-top:.3rem'>{correct} / {total} correct</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    if   pct >= 90: st.success("🏅 Outstanding!")
    elif pct >= 75: st.success("🎉 Excellent work!")
    elif pct >= 60: st.info("👍 Good effort — keep going!")
    elif pct >= 45: st.warning("📚 Review the explanations and try again.")
    else:           st.error("💡 Study the material and retake the quiz.")

    with st.expander("Detailed breakdown"):
        for i, m in enumerate(mcqs, 1):
            ua  = st.session_state.answers.get(f"q_{i}", "—")
            ok  = m["correct_answer"]
            ico = "✅" if ua == ok else "❌"
            st.markdown(f"{ico} **Q{i}** — Your: `{ua}` · Correct: `{ok}`")


# ── Export tab ────────────────────────────────────────────────────────────────

def render_export(mcqs: List[Dict], topic: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = topic[:28].replace(" ", "_")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "⬇  JSON",
            json.dumps(mcqs, indent=2, ensure_ascii=False),
            f"quiz_{slug}_{ts}.json",
            "application/json",
            use_container_width=True,
        )
    with c2:
        rows = []
        for i, m in enumerate(mcqs, 1):
            row = {"#": i, "Question": m["question"],
                   "Answer": m["correct_answer"], "Explanation": m["explanation"]}
            row.update({f"Opt {k}": v for k, v in m["options"].items()})
            rows.append(row)
        st.download_button(
            "⬇  CSV",
            pd.DataFrame(rows).to_csv(index=False),
            f"quiz_{slug}_{ts}.csv",
            "text/csv",
            use_container_width=True,
        )
    with c3:
        lines = [f"QUIZ: {topic}", f"Date: {datetime.now():%Y-%m-%d %H:%M}", f"Questions: {len(mcqs)}", "=" * 60, ""]
        for i, m in enumerate(mcqs, 1):
            lines += [f"Q{i}: {m['question']}", ""]
            lines += [f"  {k}. {v}" for k, v in m["options"].items()]
            lines += ["", f"  Answer: {m['correct_answer']}", f"  {m['explanation']}", "", "-" * 40, ""]
        st.download_button(
            "⬇  TXT",
            "\n".join(lines),
            f"quiz_{slug}_{ts}.txt",
            "text/plain",
            use_container_width=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Client ─────────────────────────────────────────────────────────────
    try:
        client = get_client()
    except ValueError as e:
        st.markdown('<div class="hero"><h1>QuizAI</h1></div>', unsafe_allow_html=True)
        st.error(str(e))
        with st.expander("Create .env file"):
            st.code(
                "GOOGLE_API_KEY=your_key_here   # https://aistudio.google.com/app/apikey\n"
                "GROQ_API_KEY=your_key_here     # https://console.groq.com/keys",
                language="bash"
            )
            if st.button("Generate .env template"):
                create_env_template()
                st.success("Created .env — add your keys and restart.")
        return

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;"
            "color:#a78bfa;margin-bottom:1.5rem'>⚡ QuizAI</p>",
            unsafe_allow_html=True
        )

        # Provider badges
        status = client.provider_status()
        badge = lambda ok: (
            "<span style='background:#14532d;color:#4ade80;padding:.15rem .5rem;"
            "border-radius:6px;font-size:.72rem;font-weight:600'>ON</span>"
            if ok else
            "<span style='background:#3b1111;color:#f87171;padding:.15rem .5rem;"
            "border-radius:6px;font-size:.72rem;font-weight:600'>OFF</span>"
        )
        st.markdown(
            f"<div style='display:flex;gap:.6rem;align-items:center;margin-bottom:1rem'>"
            f"Gemini {badge(status['gemini'])} &nbsp; Groq {badge(status['groq'])}"
            f"</div>",
            unsafe_allow_html=True
        )

        # Model
        models  = client.available_models
        m_ids   = [m.id for m in models]
        m_labels = [f"{m.name} — {m.description}" for m in models]
        sel_idx  = st.selectbox("Model", range(len(m_ids)), format_func=lambda i: m_labels[i])

        st.markdown("---")

        # Settings
        n_q   = st.slider("Questions", 1, 50, 10)
        diff  = st.select_slider("Difficulty", ["easy", "medium", "hard"], "medium")
        n_opt = st.selectbox("Options per question", [3, 4, 5], index=1)
        qpp   = st.selectbox("Per page", [3, 5, 10, 20], index=1)

        st.markdown("---")
        st.markdown(
            "<p style='font-size:.72rem;color:#444;text-align:center'>"
            "Gemini 60 req/min · Groq generous free tier</p>",
            unsafe_allow_html=True
        )

    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="hero">'
        '<h1>Generate a Quiz Instantly</h1>'
        '<p>Powered by Google Gemini &amp; Groq — enter any topic below</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Input row ───────────────────────────────────────────────────────────
    ic1, ic2 = st.columns([5, 1])
    with ic1:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. Python decorators, The Roman Empire, Quantum entanglement…",
            label_visibility="collapsed",
        )
    with ic2:
        generate = st.button("Generate ⚡", type="primary", use_container_width=True, disabled=not topic)

    # Quick-pick chips
    chips = ["Python basics", "World War II", "Cell biology", "Linear algebra",
             "Machine learning", "The Solar System", "Economics 101", "Data Structures"]
    chip_cols = st.columns(len(chips))
    for col, chip in zip(chip_cols, chips):
        with col:
            if st.button(chip, use_container_width=True, type="secondary"):
                topic = chip
                generate = True

    # ── Generation ──────────────────────────────────────────────────────────
    if generate and topic:
        try:
            client.set_model(m_ids[sel_idx])
        except ValueError as e:
            st.error(str(e))
            return

        with st.spinner(f"Generating {n_q} questions…"):
            try:
                mcqs = client.generate_mcqs(topic, n_q, diff, n_opt)
                st.session_state.mcqs       = mcqs
                st.session_state.topic      = topic
                st.session_state.difficulty = diff
                st.session_state.qpp        = qpp
                st.session_state.answers    = {}
                st.session_state.show_ans   = False
                st.session_state.submitted  = False
                st.session_state.quiz_page  = 1
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # ── Results ─────────────────────────────────────────────────────────────
    if st.session_state.mcqs:
        mcqs  = st.session_state.mcqs
        topic = st.session_state.topic

        # Header
        h1, h2, h3 = st.columns(3)
        with h1: st.metric("Questions", len(mcqs))
        with h2: st.metric("Difficulty", st.session_state.difficulty.title())
        with h3: st.metric("Answered", f"{len(st.session_state.answers)}/{len(mcqs)}")

        st.markdown("")

        tab1, tab2 = st.tabs(["📝  Quiz", "💾  Export"])
        with tab1:
            render_quiz(mcqs)
        with tab2:
            render_export(mcqs, topic)


if __name__ == "__main__":
    main()