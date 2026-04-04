import streamlit as st
import time
import re
import pandas as pd
import joblib

# -------------------------------
# LOAD AI MODEL
# -------------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="BharatFilter AI", layout="wide")

# -------------------------------
# SIDEBAR (DASHBOARD NAVIGATION)
# -------------------------------
st.sidebar.title("🇮🇳 BharatFilter AI")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "🔍 Analyze Text", "📁 File Analysis", "ℹ️ About"])

# -------------------------------
# TOXIC WORD LIST
# -------------------------------
toxic_words = [
    "hate","kill","stupid","idiot","dumb","ugly","fool","loser","trash","bitch",
    "बेवकूफ","नफरत","मार","गधा","पागल",
    "मूर्ख","द्वेष","मार","वेडा","नालायक"
]

# -------------------------------
# ALGORITHMS
# -------------------------------
def naive_search(text, pattern):
    positions = []
    n, m = len(text), len(pattern)
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    return positions

def rabin_karp(text, pattern, d=256, q=101):
    n, m = len(text), len(pattern)
    if m > n:
        return []

    p = t = 0
    h = 1
    positions = []

    for i in range(m - 1):
        h = (h * d) % q

    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    for i in range(n - m + 1):
        if p == t and text[i:i+m] == pattern:
            positions.append(i)

        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
            if t < 0:
                t += q

    return positions

# -------------------------------
# AI DETECTION
# -------------------------------
def ai_detect(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

# -------------------------------
# ANALYSIS FUNCTION
# -------------------------------
def analyze_text(user_input):
    text = user_input.lower()

    start_naive = time.time()
    naive_results = {}
    for word in toxic_words:
        pos = naive_search(text, word.lower())
        if pos:
            naive_results[word] = pos
    end_naive = time.time()

    start_rk = time.time()
    rk_results = {}
    for word in toxic_words:
        pos = rabin_karp(text, word.lower())
        if pos:
            rk_results[word] = pos
    end_rk = time.time()

    detected_words = list(set(list(naive_results.keys()) + list(rk_results.keys())))
    count = len(detected_words)

    return detected_words, count, end_naive-start_naive, end_rk-start_rk

# ===============================
# HOME PAGE
# ===============================
if menu == "🏠 Home":
    st.title("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "AI + Algorithms")
    col2.metric("Languages", "English + Hindi")
    col3.metric("Accuracy", "High")

    st.markdown("---")
    st.info("Use sidebar to navigate between features")

# ===============================
# TEXT ANALYSIS
# ===============================
elif menu == "🔍 Analyze Text":
    st.title("🔍 Text Analysis")

    user_input = st.text_area("Enter Text")

    if st.button("🚀 Analyze"):
        if user_input.strip() == "":
            st.warning("Enter text first!")
        else:
            words, count, t1, t2 = analyze_text(user_input)
            total_words = len(user_input.split())
            percentage = (count / total_words * 100) if total_words > 0 else 0

            # AI RESULT
            ai_result = ai_detect(user_input)

            st.subheader("🤖 AI Result")
            if ai_result == 1:
                st.error("Toxic Content Detected")
            else:
                st.success("Clean Content")

            # METRICS
            col1, col2, col3 = st.columns(3)
            col1.metric("Toxic Words", count)
            col2.metric("Total Words", total_words)
            col3.metric("Toxic %", f"{percentage:.2f}%")

            # RULE-BASED
            if count == 0:
                st.success("✅ No Toxic Content (Rule-Based)")
            else:
                st.error("⚠️ Toxic Content Detected (Rule-Based)")

            # HIGHLIGHT
            highlighted = user_input
            for w in words:
                highlighted = re.sub(w, f"🔴{w.upper()}", highlighted, flags=re.IGNORECASE)

            st.subheader("📝 Highlighted Text")
            st.write(highlighted)

            st.subheader("📌 Detected Words")
            st.write(words)

            # GRAPH
            st.subheader("📊 Algorithm Performance")
            st.bar_chart({"Naive": [t1], "Rabin-Karp": [t2]})

# ===============================
# FILE ANALYSIS
# ===============================
elif menu == "📁 File Analysis":
    st.title("📁 File Analysis")

    file = st.file_uploader("Upload CSV or TXT")

    if file:
        if file.name.endswith(".txt"):
            content = file.read().decode("utf-8")
            ai_result = ai_detect(content)

            if ai_result == 1:
                st.error("🤖 Toxic Content Detected")
            else:
                st.success("🤖 Clean Content")

        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)

            if "text" in df.columns:
                df["AI_Result"] = df["text"].apply(lambda x: ai_detect(str(x)))
                st.dataframe(df)
                st.success("Analysis Completed!")

# ===============================
# ABOUT PAGE
# ===============================
elif menu == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.write("""
    BharatFilter AI is a multilingual toxic content detection system.

    🔹 Uses Machine Learning (TF-IDF + Logistic Regression)  
    🔹 Implements Naive & Rabin-Karp algorithms  
    🔹 Supports text and file input  
    🔹 Provides real-time analysis  

    🎯 Goal: Safe digital communication
    """)
