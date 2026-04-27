import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader

st.set_page_config(page_title="Quiz Generator", page_icon="📚")

st.title("📚 AI Quiz Generator")

# ✅ Load better model (instruction-based)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# 📂 File Upload
file = st.file_uploader("Upload PDF", type=["pdf"])

# 🎛️ Controls
num_q = st.slider("Number of MCQs", 1, 15, 5)
difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])

# 📄 Extract Text
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

# 🎯 Generate Quiz
def generate_quiz(text, num_q, difficulty):
    prompt = f"""
You are a professional teacher.

Create exactly {num_q} {difficulty} multiple choice questions (MCQs) from the given text.

Rules:
- Each question must be clear and short
- Provide exactly 4 options (A, B, C, D)
- Only one correct answer
- Show the correct answer clearly
- Do not repeat questions

Format strictly like this:

Q1. Question here
A) Option
B) Option
C) Option
D) Option
Answer: A

Text:
{text[:1500]}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

# 🚀 Button Action
if st.button("Generate Quiz"):
    if file is None:
        st.warning("⚠️ Please upload a PDF file first.")
    else:
        text = extract_text(file)

        if len(text.strip()) == 0:
            st.error("❌ Could not extract text from PDF.")
        else:
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz(text, num_q, difficulty)

            st.success("✅ Quiz Generated!")

            st.text_area("📋 Your Quiz", quiz, height=400)

            # 📥 Download option
            st.download_button(
                label="Download Quiz",
                data=quiz,
                file_name="quiz.txt",
                mime="text/plain"
            )
