import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import os

st.set_page_config(
    page_title="Problem Difficulty Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Management ---
if 'ai_check_done' not in st.session_state:
    st.session_state.ai_check_done = False
if 'ai_valid_status' not in st.session_state:
    st.session_state.ai_valid_status = True
if 'ai_message' not in st.session_state:
    st.session_state.ai_message = ""
if 'input_cache' not in st.session_state:
    st.session_state.input_cache = ""

KEYWORDS = [
    "dp","greedy","binary search","two pointers","sliding window",
    "recursion","backtracking","divide and conquer","bitmask",
    "array","string","stack","queue","heap","priority queue",
    "hashmap","set","tree","binary tree","bst","segment tree",
    "fenwick","trie","graph","dag","linked list","disjoint set",
    "union find","bfs","dfs","shortest path","dijkstra",
    "bellman ford","floyd","mst","kruskal","prim","topological",
    "cycle","bipartite","modulo","gcd","lcm","prime","sieve",
    "combinatorics","probability","matrix","prefix sum","xor",
    "bitwise","substring","subsequence","palindrome",
    "z algorithm","kmp","hashing","simulation","implementation",
    "geometry","game theory"
]

def numeric_features(X):
    if isinstance(X, pd.DataFrame):
        text = X.iloc[:, 0]
    elif isinstance(X, pd.Series):
        text = X
    else:
        text = pd.Series(X)

    text = text.fillna("").astype(str)

    length = text.str.len().values.reshape(-1, 1)
    math_symbols = text.str.count(r"[=<>+\-*/%^]").values.reshape(-1, 1)

    keyword_counts = np.column_stack([
        text.str.contains(rf"\b{k}\b", case=False, regex=True).astype(int)
        for k in KEYWORDS
    ])

    return np.hstack([length, math_symbols, keyword_counts])

def sparse_to_dense(x):
    return x.toarray()

@st.cache_resource
def load_models():
    #stage0 = joblib.load("stage0_invalid_detector.pkl")
    stage1 = joblib.load("stage1_hard_classifier.pkl")
    HARD_T = joblib.load("hard_threshold.pkl")
    stage2 = joblib.load("stage2_easy_medium.pkl")
    regressors = joblib.load("score_regressors.pkl") 
    return stage1, HARD_T, stage2, regressors
    #return stage1, HARD_T, stage2, reg

@st.cache_data(ttl=3600)
def validate_with_ai(full_text: str, api_key: str) -> tuple[bool, str]:
    if not api_key or api_key.strip() == "":
        return True, "Skipped - no API key provided"
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key.strip())
        
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception:
            available_models = ["models/gemini-1.5-flash", "models/gemini-pro"]

        target_model = None
        priority_order = ["models/gemini-1.5-flash", "models/gemini-pro", "models/gemini-1.0-pro"]
        
        for p in priority_order:
            if p in available_models:
                target_model = p
                break
        
        if not target_model and available_models:
            target_model = available_models[0]
            
        if not target_model:
            target_model = "gemini-pro"

        model = genai.GenerativeModel(target_model)
        
        prompt = f"""You are a competitive programming expert. Determine if this is a VALID CODING PROBLEM.
        PROBLEM TEXT: "{full_text[:1500]}"
        Respond with ONLY ONE WORD: VALID or INVALID"""

        response = model.generate_content(prompt)
        
        result = response.text.strip().upper()
        is_valid = result.startswith("VALID")
        
        status = f"AI Confirmed ({target_model.split('/')[-1]}): Valid" if is_valid else "AI Alert: Content does not appear to be a coding problem."
        return is_valid, status
        
    except ImportError:
        return True, "Skipped - google-generativeai library not installed"
    except Exception as e:
        error_msg = str(e).lower()
        if "400" in error_msg or "invalid argument" in error_msg:
             return False, "Error: Invalid Gemini API Key."
        
        return True, f"AI Error (Switching to local): {error_msg[:40]}..."

# --- Main Prediction Logic Refactored for Reusability ---
def run_prediction_pipeline(full_text_input):
    loader_steps = [
        "Evaluating complexity...", 
        "Computing score...", 
        "Finalizing prediction..."
    ]
    
    local_loader = st.empty()
    
    for i, step in enumerate(loader_steps):
        with local_loader.container():
            st.markdown(f"""
                <div class="loader-container">
                    <div class="loader {'loader-stopped' if i == 4 else ''}"></div>
                    <div class="loader-text">{step}</div>
                </div>
            """, unsafe_allow_html=True)
        time.sleep(0.5) 
    
    local_loader.empty()
    st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)

    try:
        stage1, HARD_T, stage2, regressors = load_models()

        X = pd.DataFrame({"full_text": [full_text_input]})
        
        # Determine dominant topic for visualization
        dominant_topic = "algorithm data structure"
        for k in KEYWORDS:
            if k in full_text_input.lower():
                dominant_topic = k
                break # Just take the first hit for simplicity
        
        #valid_pred = stage0.predict(X)[0]

        hard_prob = stage1.predict_proba(X)[0, 1]
        is_hard = hard_prob >= HARD_T
        
        if is_hard:
            level = "hard"
        else:
            level = stage2.predict(X)[0]
        
        #score = float(reg.predict(X)[0])
        #score = max(1.0, min(10.0, score))
        # NEW (uses class-specific regressors correctly)
        #raw_score = regressors[level].predict(X)[0]
        raw_score = regressors.predict(X)[0]
        score = np.clip(raw_score, 1.0, 10.0)
        st.markdown('<h2 class="result-title">Prediction Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <div class="label">Predicted Difficulty</div>
            """, unsafe_allow_html=True)
            
            level_html = f'<div class="value {level}">{level.upper()}</div>'
            st.markdown(level_html, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <div class="label">Difficulty Score</div>
            """, unsafe_allow_html=True)
            
            score_html = f'<div class="score">{score:.1f}</div>'
            st.markdown(score_html, unsafe_allow_html=True)
            
            st.markdown("""
                <div class="label" style="font-size: 13px; color: #9ca3af;">
                    1 = Easiest, 10 = Hardest
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.success(f"Analysis complete! Predicted {level.upper()} (Score: {score:.1f}/10)")
        
    except FileNotFoundError:
        st.error("Missing model files! Please ensure all .pkl files are in the same directory.")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'JetBrains Mono', monospace; }

body, h1, h2, h3, p { font-family: 'JetBrains Mono', monospace !important; }

.block {
    background: linear-gradient(145deg, #111827, #1f2937);
    border-radius: 20px;
    padding: 28px;
    margin: 20px 0;
    border: 1px solid rgba(75, 85, 99, 0.3);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

.label {
    color: #9ca3af;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.value {
    font-size: 48px;
    font-weight: 800;
    margin: 8px 0;
    line-height: 1.1;
}

.score {
    font-size: 64px;
    font-weight: 900;
    letter-spacing: -0.02em;
}

.easy { 
    background: linear-gradient(135deg, #4ade80, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.medium { 
    background: linear-gradient(135deg, #facc15, #eab308);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hard { 
    background: linear-gradient(135deg, #f87171, #ef4444);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-card {
    background: linear-gradient(145deg, #0f172a, #1e293b);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(59, 130, 246, 0.2);
    margin: 12px 0;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

.dotted-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    margin: 20px 0;
}

.loader-container {
    text-align: center;
    padding: 40px;
}

.loader {
    width: 60px;
    height: 60px;
    border: 4px solid rgba(59, 130, 246, 0.2);
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 20px;
}

.loader-stopped {
    animation: none !important;
    border-top-color: #10b981 !important;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loader-text {
    font-size: 18px;
    font-weight: 600;
    color: #3b82f6;
    margin: 0;
}

.main-title {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    color: #3b82f6 !important;
    text-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
    letter-spacing: -0.02em !important;
    margin-bottom: 10px !important;
    text-align: center !important;
}

.subtitle {
    font-size: 1.3rem !important;
    color: #6b7280 !important;
    font-weight: 500 !important;
    text-align: center !important;
    margin: 0 !important;
}

.result-title {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #3b82f6 !important;
    text-align: center !important;
    margin: 40px 0 25px 0 !important;
    text-shadow: 0 0 10px rgba(59, 130, 246, 0.2);
}

.sidebar-tips {
    color: #60a5fa !important;
    font-weight: 500;
}

.status-valid {
    color: #4ade80;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    background: rgba(74, 222, 128, 0.1);
    border-radius: 8px;
    margin-bottom: 20px;
}
.status-invalid {
    color: #ef4444;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 8px;
    margin-bottom: 20px;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* --- UPDATED BUTTON STYLES --- */

/* 1. Main "Analyze" Button (Teal / Aqua) */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #14B8A6, #0D9488) !important; /* Primary Teal */
    border: 1px solid rgba(20, 184, 166, 0.5) !important;             /* Teal Border */
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px -1px rgba(20, 184, 166, 0.2);
}

div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(20, 184, 166, 0.4); /* Teal Glow */
    border-color: #5EEAD4 !important;                     /* Bright Accent Border */
    background: linear-gradient(135deg, #0D9488, #115e59) !important; /* Darker on hover */
}

/* 2. "Proceed Anyway" Button (Red/Warning) */
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 2px solid #ef4444 !important; /* Red Border */
    color: #ef4444 !important;            /* Red Text */
    font-weight: 700 !important;
    border-radius: 12px !important;
}

div.stButton > button[kind="secondary"]:hover {
    background: rgba(239, 68, 68, 0.1) !important;
    border-color: #f87171 !important;
}

div.stButton > button:active {
    transform: translateY(0);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 40px; padding: 20px;">
    <h1 class="main-title">Problem Difficulty Predictor</h1>
    <p class="subtitle">Predict coding problem difficulty in 3 classes + score (1-10)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-tips">', unsafe_allow_html=True)
    st.markdown("### AI Configuration")
    api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Enter Google Gemini key for AI validation")
    
    st.markdown("---")
    st.markdown("### Usage Tips")
    st.markdown("**Works best with** complete coding problems")
    st.markdown("**Avoid** random text or non-coding content")
    st.markdown("**3 Classes:** Easy | Medium | Hard")
    st.markdown("**Score:** 1 (Easiest) → 10 (Hardest)")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="block" style="margin-top: 0;">', unsafe_allow_html=True)

st.markdown('<div class="label" style="margin-bottom: 16px;">Problem Details</div>', unsafe_allow_html=True)

problem_text = st.text_area(
    "Paste your complete problem statement here...",
    height=200,
    placeholder="""Example:
You are given an array of integers nums and an integer target.
Return indices of the two numbers such that they add up to target.

Input Format:
First line: N (number of elements)
...""",
    help="Include problem description, constraints, input/output format"
)

input_format = st.text_area(
    "Input Format (optional)",
    height=100,
    placeholder="Detailed input specification..."
)

output_format = st.text_area(
    "Output Format (optional)", 
    height=100,
    placeholder="Detailed output specification..."
)

st.markdown('</div>', unsafe_allow_html=True)

# Construct full text
current_full_text = (
    problem_text.strip() + " " +
    (input_format or "").strip() + " " +
    (output_format or "").strip()
)

# Button Logic
analyze_col, empty_col = st.columns([1, 2])
with analyze_col:
    # --- CHANGED: Added type="primary" to link to Teal CSS ---
    analyze_clicked = st.button("Analyze Difficulty", use_container_width=True, type="primary")

# Validation and Prediction Flow
if analyze_clicked:
    if not problem_text.strip():
        st.error("Problem description is required!")
        st.stop()

        
    st.session_state.input_cache = current_full_text
    
    # 1. AI Validation Step
    loader_container = st.empty()
    if api_key:
        with loader_container.container():
            st.markdown("""
                <div class="loader-container">
                    <div class="loader"></div>
                    <div class="loader-text">AI Guardrail: Validating Input (Gemini)...</div>
                </div>
            """, unsafe_allow_html=True)
        
        is_valid_ai, ai_msg = validate_with_ai(current_full_text, api_key)
        loader_container.empty()
        
        st.session_state.ai_valid_status = is_valid_ai
        st.session_state.ai_message = ai_msg
        st.session_state.ai_check_done = True
    else:
        # No API Key, skip validation
        st.session_state.ai_valid_status = True
        st.session_state.ai_check_done = True
        st.session_state.ai_message = ""

# Handling Logic after click (or if validation failed previously)
if st.session_state.ai_check_done and st.session_state.input_cache:
    
    # CASE A: Validation Failed
    if not st.session_state.ai_valid_status:
        st.markdown('<div class="dotted-line"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-invalid">{st.session_state.ai_message}</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align:center; color: #9ca3af; margin-bottom: 15px;">
            The AI suggests this may not be a valid coding problem. You can choose to proceed anyway.
        </div>
        """, unsafe_allow_html=True)
        
        col_warn1, col_warn2, col_warn3 = st.columns([1,2,1])
        with col_warn2:
            # --- CHANGED: Added type="secondary" to link to Red CSS ---
            if st.button("Proceed Anyway", type="secondary", use_container_width=True):
                 run_prediction_pipeline(st.session_state.input_cache)
                 st.session_state.ai_check_done = False # Reset
        
    # CASE B: Validation Passed or Skipped
    else:
        if st.session_state.ai_message:
             st.markdown(f'<div class="status-valid">{st.session_state.ai_message}</div>', unsafe_allow_html=True)
             time.sleep(1)
        
        run_prediction_pipeline(st.session_state.input_cache)
        st.session_state.ai_check_done = False # Reset
