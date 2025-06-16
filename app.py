import os
import streamlit as st
import pdfplumber
import docx
import pandas as pd
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="📚 Ask Your Documents", layout="wide")
st.title("📚 Multi-Format Document Chatbot")

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=None)
        return "\n".join([df[sheet].to_string() for sheet in df])
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

@st.cache_resource
def load_documents():
    all_chunks = []
    file_dir = "./docs"
    files = [f for f in os.listdir(file_dir) if f.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls', '.txt'))]

    for file_name in files:
        full_path = os.path.join(file_dir, file_name)
        raw_text = extract_text_from_file(full_path)
        chunks = raw_text.split("\n\n")
        all_chunks.extend(chunks)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return all_chunks, index, model

chunks, index, model = load_documents()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about the documents")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    q_embedding = model.encode([user_input])
    D, I = index.search(np.array(q_embedding), k=3)
    context = "\n\n".join([chunks[i] for i in I[0]])

    messages = [
        {"role": "system", "content": f"Use only the following context from the documents to answer:\n\n{context}"},
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    answer = response.choices[0].message["content"]
    st.session_state.chat_history.append(("assistant", answer))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
