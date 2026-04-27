import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import time

st.set_page_config(page_title="Muzi AI Quiz Studio", page_icon="📚")

st.title("📚 Muzi AI Quiz Studio")
st.subheader("NUST Balochistan Campus Edition")

# ✅ Load model with GPU support check
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# 📂 File Upload
file = st.file_uploader("Upload PDF (Lecture/Notes)", type=["pdf"])

# 🎛️ Controls
col1, col2 = st.columns(2)
with col1:
    num_q = st.slider("Number of MCQs", 1, 10, 5)
with col2:
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

# 📄 Extract Text Logic
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages[:5]: # First 5 pages for better context
        content = page.extract_text()
        if content:
            text += content
    return text

# 🎯 Single MCQ Generator (Looping for Stability)
def generate_single_mcq(context, q_num, difficulty):
    prompt = f"Generate one {difficulty} MCQ from this text. \nContext: {context[:800]} \nFormat: Q{q_num}. [Question] \nA) [Opt] B) [Opt] C) [Opt] D) [Opt] \nAnswer: [Key]"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs, 
        max_length=256, 
        do_sample=True, 
        temperature=0.7,
        repetition_penalty=1.2
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 🚀 Generate Button Action
if st.button("🚀 Generate Quiz"):
    if file is None:
        st.warning("⚠️ Muzi bhai, pehle PDF toh upload karein!")
    else:
        text = extract_text(file)
        
        if len(text.strip()) < 100:
            st.error("❌ PDF mein text bohot kam hai.")
        else:
            full_quiz = ""
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Generating one by one for high quality
            for i in range(1, num_q + 1):
                status_text.text(f"Generating Question {i} of {num_q}...")
                # Random chunk selection for variety
                q_text = generate_single_mcq(text, i, difficulty)
                full_quiz += q_text + "\n\n"
                progress_bar.progress(i / num_q)
            
            st.success("✅ Quiz Generated Successfully!")
            st.text_area("📋 Final Quiz Output", full_quiz, height=400)

            # 📥 Download button
            st.download_button(
                label="📥 Download Quiz (TXT)",
                data=full_quiz,
                file_name=f"muzi_quiz_{difficulty}.txt",
                mime="text/plain"
            )
