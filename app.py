
import os
import streamlit as st
import pdfplumber
import docx
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="📚 Ask Your Documents", layout="wide")
st.title("📚 Multi-Format Document Chatbot")

# Extract text from supported file types
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=None)
        return "\n".join(df[sheet].to_string() for sheet in df)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Smarter chunking: preserve paragraph structure and merge small chunks
def smart_chunking(text, max_len=700):
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    combined = []
    temp = ""

    for para in paragraphs:
        if len(temp) + len(para) < max_len:
            temp += " " + para
        else:
            combined.append(temp.strip())
            temp = para
    if temp:
        combined.append(temp.strip())

    return combined

# Load and embed all document content with filenames
@st.cache_resource
def load_documents():
    docs_folder = "./docs"
    file_chunks = []  # List of (filename, text chunk)

    for filename in os.listdir(docs_folder):
        if filename.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls', '.txt')):
            full_path = os.path.join(docs_folder, filename)
            raw_text = extract_text_from_file(full_path)
            if raw_text:
                for chunk in smart_chunking(raw_text):
                    file_chunks.append((filename, chunk))

    texts = [chunk for (_, chunk) in file_chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return file_chunks, index, model

file_chunks, index, model = load_documents()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about the documents")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    query_embedding = model.encode([user_input])
    _, indices = index.search(np.array(query_embedding), k=5)  # top 5 chunks

    selected_chunks = [file_chunks[i] for i in indices[0]]  # [(filename, chunk), ...]
    context = "\n\n".join([f"[{fn}]\n{text}" for fn, text in selected_chunks])

    messages = [
        {
            "role": "system",
            "content": f"You’re a sharp, slightly sassy assistant who gives clear, business-casual answers with a touch of dry humor. Don’t ramble — be efficient, maybe crack a subtle joke, and stick strictly to the info below:\n\n{context}"
        },
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    answer = response.choices[0].message.content
    st.session_state.chat_history.append(("assistant", answer))

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
