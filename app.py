import os
import streamlit as st
import pdfplumber
import docx
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from collections import defaultdict

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="Mike Dpt", layout="wide")
st.title("🏅 Mike Dpt NBC Olympic Audio Chatbot 🏅")

# Load supported documents (only ones that successfully open)
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
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
    except Exception as e:
        print(f"❌ Skipping file {file_path}: {e}")
    return ""

# Chunk text into readable chunks
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

# Cache document loading
@st.cache_resource
def load_documents():
    file_chunks = []  # (doc_number, filename, chunk)
    valid_files = []

    raw_docs = sorted([
        f for f in os.listdir("./docs")
        if f.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls', '.txt'))
    ])

    for f in raw_docs:
        full_path = os.path.join("./docs", f)
        text = extract_text_from_file(full_path)
        if text:
            valid_files.append(f)

    doc_map = {str(i + 1): fname for i, fname in enumerate(valid_files)}

    for num, filename in doc_map.items():
        full_path = os.path.join("./docs", filename)
        raw_text = extract_text_from_file(full_path)
        for chunk in smart_chunking(raw_text):
            file_chunks.append((num, filename, chunk))

    texts = [chunk for (_, _, chunk) in file_chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return file_chunks, index, model, doc_map

file_chunks, index, model, doc_map = load_documents()

# Sidebar
with st.sidebar:
    st.subheader("📂 Available Content")
    for num, fname in doc_map.items():
        st.markdown(f"**{num}.** `{fname}`")

# Track chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Do you have questions about the available content?")
file_qs = ["what documents do you have", "list all documents", "what files are loaded", "what files do you have"]
intercepted = user_input and any(q in user_input.lower() for q in file_qs)

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    if intercepted:
        reply = "**Here are the documents I can reference:**\n\n" + "\n".join(
            f"**{num}.** {name}" for num, name in doc_map.items()
        )
        st.session_state.chat_history.append(("assistant", reply))

    else:
        target_doc_num = None
        for word in user_input.lower().split():
            if word.isdigit() and word in doc_map:
                if f"document {word}" in user_input.lower():
                    target_doc_num = word
                    break

        if target_doc_num:
            relevant_chunks = [(n, f, c) for (n, f, c) in file_chunks if n == target_doc_num]
            query_embedding = model.encode([user_input])
            chunk_embeddings = model.encode([chunk for (_, _, chunk) in relevant_chunks])
            chunk_index = faiss.IndexFlatL2(chunk_embeddings[0].shape[0])
            chunk_index.add(np.array(chunk_embeddings))
            _, local_indices = chunk_index.search(query_embedding, k=min(6, len(relevant_chunks)))
            selected_chunks = [relevant_chunks[i] for i in local_indices[0]]
        else:
            query_embedding = model.encode([user_input])
            _, global_indices = index.search(np.array(query_embedding), k=8)
            selected_chunks = [file_chunks[i] for i in global_indices[0]]

        grouped = defaultdict(list)
        for doc_num, fname, chunk in selected_chunks:
            grouped[(doc_num, fname)].append(chunk)

        best_doc = max(grouped.items(), key=lambda x: len(x[1]))
        (doc_num, filename), best_chunks = best_doc
        context = "\n\n".join(best_chunks)

        summary_msg = f"💡 Responding using **Document {doc_num}: {filename}**\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a sharp, sassy, business-casual assistant. Answer using only this content from Document {doc_num} ({filename}). "
                    f"If helpful, summarize with bullets or clarity:\n\n{context}"
                )
            },
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gpt-4",  # <-- Change to gpt-4 here if you like
            messages=messages
        )
        answer = summary_msg + response.choices[0].message.content
        st.session_state.chat_history.append(("assistant", answer))

# Render history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
