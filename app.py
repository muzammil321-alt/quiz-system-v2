import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

st.title("📚 Quiz Generator")

# Load model (once)
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# File upload
file = st.file_uploader("Upload PDF", type=["pdf"])

# Inputs
num_q = st.slider("Number of MCQs", 1, 20, 5)
difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

if st.button("Generate Quiz"):
    if file is None:
        st.warning("Please upload a file first.")
    else:
        text = extract_text(file)

        prompt = f"""
        Generate {num_q} {difficulty} multiple choice questions (MCQs).
        Each question must have 4 options (A, B, C, D) and correct answer.

        Text:
        {text[:1000]}
        """

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=500)

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.text_area("Generated Quiz", result, height=400)
