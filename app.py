import streamlit as st
import pickle
import re
import time
import numpy as np

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Newsreader:ital,wght@0,400;0,600;1,400;1,600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #f7f9fb !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #191c1e !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; visibility: hidden !important; }

.block-container {
    padding: 4rem 1.5rem 4rem !important;
    max-width: 720px !important;
}

/* ── Top accent bar ── */
.top-bar {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 4px;
    background: linear-gradient(to right, #00685f, #006a63, #006b2c);
    opacity: 0.5; z-index: 9999;
}

/* ── Hero ── */
.hero {
    text-align: center;
    display: flex; flex-direction: column;
    align-items: center; gap: 1.5rem;
    margin-bottom: 3rem;
}
.hero-badge {
    display: inline-flex; align-items: center;
    padding: 6px 16px; border-radius: 999px;
    background: #99efe5; color: #006f67;
    font-size: 12px; font-weight: 800;
    letter-spacing: 0.15em; text-transform: uppercase;
    border: 1px solid rgba(0,104,95,0.1);
}
.hero-title {
    font-family: 'Newsreader', serif;
    font-size: 3.75rem; font-weight: 600;
    color: #191c1e; line-height: 1.1;
    letter-spacing: -0.02em;
}
.hero-title em { font-style: italic; font-weight: 400; color: #00685f; }
.hero-sub {
    font-size: 1.125rem; color: #3d4947;
    max-width: 540px; line-height: 1.7;
}

/* ── Stats ── */
.stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 3rem;
}
.stat-card {
    background: #ffffff; border-radius: 0.75rem;
    padding: 1.25rem; text-align: center;
}
.stat-label {
    display: block; font-size: 10px; font-weight: 800;
    color: #bcc9c6; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 6px;
}
.stat-value {
    display: block; font-size: 1.25rem;
    font-weight: 700; color: #00685f; line-height: 1;
}

/* ── Input card ── */
.input-section {
    background: #ffffff; border-radius: 0.75rem;
    padding: 2rem; margin-bottom: 3rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.input-section-label {
    font-size: 11px; font-weight: 800; color: #3d4947;
    letter-spacing: 0.2em; text-transform: uppercase;
    margin-bottom: 0.75rem; display: block;
}

textarea {
    background: #f2f4f6 !important;
    border: none !important;
    border-radius: 0.75rem !important;
    padding: 1.5rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    color: #191c1e !important;
    line-height: 1.6 !important;
}
textarea:focus {
    box-shadow: 0 0 0 2px rgba(0,104,95,0.2) !important;
    outline: none !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00685f, #008378) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 1rem 2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    margin-top: 1rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

/* ── Result FAKE ── */
.result-fake {
    background: #ffdad6;
    border: 1px solid rgba(186,26,26,0.1);
    border-radius: 0.75rem;
    padding: 2rem;
    text-align: center;
    display: flex; flex-direction: column;
    align-items: center; gap: 1rem;
    margin-bottom: 3rem;
    animation: fadeUp 0.35s ease;
}
.result-fake-label {
    font-family: 'Newsreader', serif;
    font-size: 1.875rem; font-weight: 700;
    color: #ba1a1a;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}

/* ── Result REAL ── */
.result-real {
    background: #7ffc97;
    border: 1px solid rgba(0,107,44,0.1);
    border-radius: 0.75rem;
    padding: 2rem;
    text-align: center;
    display: flex; flex-direction: column;
    align-items: center; gap: 1rem;
    margin-bottom: 3rem;
    animation: fadeUp 0.35s ease;
}
.result-real-label {
    font-family: 'Newsreader', serif;
    font-size: 1.875rem; font-weight: 700;
    color: #005320;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}

.result-icon { font-size: 3rem; }
.result-conf-fake { font-size: 0.875rem; font-weight: 500; color: #93000a; opacity: 0.8; }
.result-conf-real { font-size: 0.875rem; font-weight: 500; color: #005320; opacity: 0.8; }

.conf-track-fake {
    width: 100%; height: 6px; border-radius: 999px;
    overflow: hidden; background: rgba(186,26,26,0.1);
    max-width: 300px;
}
.conf-fill-fake { height: 100%; border-radius: 999px; background: #ba1a1a; }

.conf-track-real {
    width: 100%; height: 6px; border-radius: 999px;
    overflow: hidden; background: rgba(0,83,32,0.1);
    max-width: 300px;
}
.conf-fill-real { height: 100%; border-radius: 999px; background: #005320; }

/* ── Verdict box — WHITE background matching Stitch screenshot ── */
.verdict-box {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.06);
    border-radius: 0.75rem;
    padding: 2rem;
    margin-bottom: 3rem;
    display: flex; flex-direction: column; gap: 1.5rem;
    animation: fadeUp 0.4s ease;
}
.verdict-section { display: flex; flex-direction: column; gap: 0.75rem; }
.verdict-heading {
    font-size: 0.75rem; font-weight: 800;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #191c1e;
}
/* Both paragraphs in plain Plus Jakarta Sans — matching Stitch screenshot */
.verdict-body {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.95rem; color: #3d4947; line-height: 1.7;
}

/* ── How it works ── */
.how-section {
    display: flex; flex-direction: column; gap: 2rem;
    margin-bottom: 3rem;
}
.how-title {
    font-family: 'Newsreader', serif;
    font-size: 1.5rem; font-weight: 600;
    color: #191c1e; text-align: center;
}
.how-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
}
.how-card {
    background: #ffffff; border-radius: 0.75rem;
    padding: 1.5rem;
    display: flex; flex-direction: column; gap: 1rem;
}
.how-num {
    width: 32px; height: 32px; border-radius: 50%;
    background: #00685f; color: #ffffff;
    font-size: 12px; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.how-card-title {
    font-weight: 700; font-size: 0.875rem;
    color: #191c1e; line-height: 1.3;
}
.how-card-desc {
    font-size: 0.75rem; color: #3d4947; line-height: 1.6;
}

/* ── Footer ── */
.footer {
    border-top: 1px solid #f1f5f9;
    padding: 3rem 0 1rem;
    text-align: center;
    display: flex; flex-direction: column;
    align-items: center; gap: 1rem;
}
.footer p { font-size: 0.875rem; color: #64748b; letter-spacing: 0.025em; }
.footer-nav { display: flex; gap: 1.5rem; }
.footer-nav a {
    color: #00685f; font-weight: 600;
    font-size: 0.875rem; text-decoration: none;
}

.stAlert { border-radius: 0.75rem !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>

<div class="top-bar"></div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">NLP · Machine Learning · Real-Time</div>
    <h1 class="hero-title">Is this news <em>real?</em></h1>
    <p class="hero-sub">
        Utilizing state-of-the-art PAC classification and TF-IDF vectorization
        to discern editorial veracity in seconds.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <span class="stat-label">Accuracy</span>
        <span class="stat-value">97.06%</span>
    </div>
    <div class="stat-card">
        <span class="stat-label">F1 Score</span>
        <span class="stat-value">0.9715</span>
    </div>
    <div class="stat-card">
        <span class="stat-label">Articles</span>
        <span class="stat-value">72,134</span>
    </div>
    <div class="stat-card">
        <span class="stat-label">Features</span>
        <span class="stat-value">50K</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-section"><span class="input-section-label">Paste your news text below</span>', unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    placeholder="e.g. 'Scientists confirm vaccine causes 5G activation in bloodstream...' or paste a full article excerpt.",
    height=150,
    label_visibility="collapsed",
)

detect_clicked = st.button("🔍  Detect Now", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if detect_clicked:
    if not user_input.strip():
        st.warning("Please paste some news text first.")
    elif len(user_input.strip().split()) < 5:
        st.warning("Please enter at least 5 words for a reliable prediction.")
    else:
        with st.spinner("Analysing..."):
            time.sleep(0.5)

        cleaned = clean_text(user_input)
        vec     = vectorizer.transform([cleaned])
        pred    = model.predict(vec)[0]
        score   = model.decision_function(vec)[0]
        conf    = round(float(1 / (1 + np.exp(-abs(score)))) * 100, 1)
        bar_w   = int(conf)

        if pred == 1:  # FAKE
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-icon">🚨</div>
                <div>
                    <div class="result-fake-label">Fake News</div>
                    <div class="result-conf-fake">Model confidence: {conf}%</div>
                </div>
                <div class="conf-track-fake">
                    <div class="conf-fill-fake" style="width:{bar_w}%"></div>
                </div>
            </div>
            <div class="verdict-box">
                <div class="verdict-section">
                    <div class="verdict-heading">What this means:</div>
                    <div class="verdict-body">
                        The linguistic patterns identified in this text align significantly
                        with historical datasets of misinformation. This includes high emotional
                        variance, clickbait syntactical structures, and unverifiable claims.
                    </div>
                </div>
                <div class="verdict-section">
                    <div class="verdict-heading">What to do:</div>
                    <div class="verdict-body">
                        Cross-reference this claim with established fact-checking bureaus
                        (e.g., Reuters, AP News). Avoid sharing this content on social platforms
                        until third-party verification is secured.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:  # REAL
            st.markdown(f"""
            <div class="result-real">
                <div class="result-icon">✅</div>
                <div>
                    <div class="result-real-label">Real News</div>
                    <div class="result-conf-real">Model confidence: {conf}%</div>
                </div>
                <div class="conf-track-real">
                    <div class="conf-fill-real" style="width:{bar_w}%"></div>
                </div>
            </div>
            <div class="verdict-box">
                <div class="verdict-section">
                    <div class="verdict-heading">What this means:</div>
                    <div class="verdict-body">
                        The writing style and vocabulary of this text align with verified
                        journalism in the WELFake training dataset of 72,134 articles.
                        Language patterns are consistent with credible editorial reporting.
                    </div>
                </div>
                <div class="verdict-section">
                    <div class="verdict-heading">Reminder:</div>
                    <div class="verdict-body">
                        No model is 100% accurate. Always verify important news with
                        multiple trusted sources before sharing or acting on it.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-section">
    <div class="how-title">How it works</div>
    <div class="how-grid">
        <div class="how-card">
            <div class="how-num">1</div>
            <div class="how-card-title">TF-IDF Vectorisation</div>
            <div class="how-card-desc">Numerical statistics that reflect how important a word is to a document in a collection or corpus.</div>
        </div>
        <div class="how-card">
            <div class="how-num">2</div>
            <div class="how-card-title">PAC Classification</div>
            <div class="how-card-desc">Passive-Aggressive Algorithms remain passive for correct predictions and aggressive for any miscalculations.</div>
        </div>
        <div class="how-card">
            <div class="how-num">3</div>
            <div class="how-card-title">Confidence Score</div>
            <div class="how-card-desc">Probabilistic weighting based on vector distance from known factual news benchmarks.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>Built by Gideon Zion Swer</p>
    <div class="footer-nav">
        <a href="https://github.com/GideonZionSwer/fake-news-detector" target="_blank">Model</a>
        <a href="https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification" target="_blank">Dataset</a>
        <a href="https://github.com/GideonZionSwer/fake-news-detector" target="_blank">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)
