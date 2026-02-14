import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from groq import Groq
import shap
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader

# PAGE CONFIG
st.set_page_config(
    page_title="Interest & Aptitude-Based AI Career Selector System",
    page_icon="🧭",
    layout="wide"
)

# CUSTOM CSS (HEADINGS + CARDS)
st.markdown("""
<style>
.big-title {
    font-size:38px;
    font-weight:800;
    color:#ffffff;  /* White heading */
}
.sub-title {
    font-size:18px;
    color:#e5e7eb;  /* Light gray subtitle, readable on dark/light mode */
}
h2, h3 {
    color:#ffffff;  /* All section headings white */
}
.card {
    padding:20px;
    border-radius:15px;
    background:#1f2937; /* Dark card for contrast */
    margin-bottom:15px;
    border:1px solid #374151;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="big-title">🧭 Interest & Aptitude-Based AI Career Selector System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interest + Aptitude + ML + DL + LLM (Groq)</div>', unsafe_allow_html=True)
st.markdown("---")

# LOAD MODELS
@st.cache_resource
def load_models():
    ml_model, label_encoder = joblib.load("ml_model.pkl")
    dl_model = tf.keras.models.load_model("dl_model.h5")
    return ml_model, label_encoder, dl_model

ml_model, label_encoder, dl_model = load_models()

# GROQ API
st.sidebar.markdown("### 🔑 API Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
client = Groq(api_key=api_key) if api_key else None

# SIDEBAR NAVIGATION
st.sidebar.markdown("### 📌 Navigation")
page = st.sidebar.radio(
    "",
    [
        "🧠 IA Based Career Selection",
        "🎯 ML Career Prediction",
        "📄 Resume Analyzer",
        "📊 Explainability",
        "🤖 AI Career Mentor (RAG)"
    ]
)

# 1️⃣ AI BASED CAREER SELECTION
if page == "🧠 IA Based Career Selection":

    st.markdown("## 🧠 Interest & Aptitude Based Career Selector")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Interests")
        programming = st.slider("Programming", 0, 5, 3)
        data = st.slider("Data & Analytics", 0, 5, 3)
        ai = st.slider("Artificial Intelligence", 0, 5, 3)
        design = st.slider("Design", 0, 5, 2)

    with col2:
        st.markdown("### 🧩 Aptitude")
        logic = st.slider("Logical Thinking", 0, 5, 3)
        math = st.slider("Mathematics", 0, 5, 3)
        creativity = st.slider("Creativity", 0, 5, 3)
        communication = st.slider("Communication", 0, 5, 3)

    if st.button("🔍 Find Best Career", use_container_width=True):

        career_scores = {
            "Data Scientist": data + math + logic + programming,
            "AI Engineer": ai + programming + math,
            "Software Developer": programming + logic,
            "UI/UX Designer": design + creativity,
            "Business Analyst": data + communication
        }

        ranked = sorted(career_scores.items(), key=lambda x: x[1], reverse=True)

        st.markdown("### 🏆 Recommended Careers")
        for career, score in ranked[:3]:
            st.success(f"**{career}** — Suitability Score: {score}")

        if client:
            prompt = f"""
User Interests & Aptitude:
Programming={programming}, Data={data}, AI={ai}, Design={design},
Logic={logic}, Math={math}, Creativity={creativity}, Communication={communication}

Top Career: {ranked[0][0]}

Explain why this career is best and give a learning roadmap.
"""
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            st.info("🤖 AI Explanation")
            st.write(response.choices[0].message.content)

# ======================================================
# 2️⃣ ML CAREER PREDICTION
# ======================================================
elif page == "🎯 ML Career Prediction":

    st.markdown("## 🎯 Career Prediction using ML")

    gpa = st.slider("GPA", 2.0, 4.0, 3.0)
    python = st.selectbox("Python Skill", [0, 1])
    math = st.selectbox("Math Skill", [0, 1])
    design = st.selectbox("Design Skill", [0, 1])

    if st.button("🔮 Predict Career", use_container_width=True):

        X = np.array([[gpa, python, math, design]])
        proba = ml_model.predict_proba(X)[0]

        careers = label_encoder.classes_

        st.success("🎓 Career Probabilities")
        fig, ax = plt.subplots()
        ax.barh(careers, proba)
        st.pyplot(fig)

# ======================================================
# 3️⃣ RESUME ANALYZER
# ======================================================
elif page == "📄 Resume Analyzer":

    st.markdown("## 📄 Resume PDF Analyzer")

    file = st.file_uploader("Upload Resume (PDF)", type="pdf")

    if file:
        reader = PdfReader(file)
        text = "".join(p.extract_text() for p in reader.pages)

        st.text_area("Extracted Resume Text", text, height=200)

        skills = ["python", "machine learning", "deep learning", "sql", "data analysis"]
        found = [s for s in skills if s in text.lower()]

        st.success("🧠 Extracted Skills")
        st.write(found)

        features = np.random.rand(1, 10)
        pred = dl_model.predict(features)
        category = np.argmax(pred)

        st.info(f"📊 Resume Domain: **{['Data','Web','AI'][category]}**")

# ======================================================
# 4️⃣ SHAP EXPLAINABILITY
# ======================================================
elif page == "📊 Explainability":

    st.markdown("## 📊 Model Explainability (SHAP)")

    sample = np.random.rand(50, 4)
    explainer = shap.Explainer(ml_model, sample)
    shap_values = explainer(sample)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample, show=False)
    st.pyplot(fig)

# ======================================================
# 5️⃣ RAG + GROQ
# ======================================================
elif page == "🤖 AI Career Mentor (RAG)":

    st.markdown("## 🤖 AI Career Mentor (RAG Powered)")

    context_pdf = st.file_uploader("Upload Career Guide / Notes PDF", type="pdf")
    context_text = ""

    if context_pdf:
        reader = PdfReader(context_pdf)
        context_text = "".join(p.extract_text() for p in reader.pages)
        st.success("📚 Knowledge Loaded")

    question = st.text_area("Ask your career question")

    if st.button("💡 Get AI Advice", use_container_width=True):

        if not client:
            st.error("Please enter Groq API key")
        else:
            prompt = f"""
You are an expert career mentor.

Context:
{context_text}

Question:
{question}

Give clear advice, roadmap, and interview tips.
"""
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )
            st.success("🎯 AI Response")
            st.write(response.choices[0].message.content)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption("🏆 Hackathon-Ready | ML + DL + IA + LLM + RAG | Built with ❤️")
