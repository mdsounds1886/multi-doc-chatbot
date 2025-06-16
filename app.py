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

# Set up OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit UI config
st.set_page_config(page_title="📚 Ask Your Documents", layout="wide")
st.title("📚 Multi-Format Document Chatbot")

# Sidebar showing list of available documents
with st.sidebar:
    st.subheader("📂 Available Documents")
    all_docs = [f for f in os.listdir("./docs") if f.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls', '.txt'))]
    for doc in sorted(all_docs):
        st.markdown(f"- `{doc}`")

# Function to extract raw text from supported file types
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

# Function to smartly chunk long documents for embedding
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

# Cache-loaded document chunks and embeddings
@st.cache_resource
def load_documents():
    docs_folder = "./docs"
    file_chunks = []  # (filename, chunk)
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

# Load chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user question
user_input = st.chat_input("Ask a question about the documents")

# Check if the user is asking about the files directly
file_qs = ["what documents do you have", "list all documents", "what files are loaded", "what files do you have"]
intercepted = user_input and any(q in user_input.lower() for q in file_qs)

# Handle new user input
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    if intercepted:
        doc_list_str = "\n".join(f"- {doc}" for doc in sorted(all_docs))
        reply = f"Here are the documents I can reference:\n\n{doc_list_str}"
        st.session_state.chat_history.append(("assistant", reply))
    else:
        query_embedding = model.encode([user_input])
        _, indices = index.search(np.array(query_embedding), k=8)

        selected_chunks = [file_chunks[i] for i in indices[0]]
        grouped = defaultdict(list)
        for filename, chunk in selected_chunks:
            grouped[filename].append(chunk)

        best_file = max(grouped.items(), key=lambda x: len(x[1]))
        context = "\n\n".join(best_file[1])
        filename = best_file[0]

        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant answering questions based strictly on the following content "
                    f"from the document [{filename}]. Be concise, clear, and if helpful, summarize in bullet points or sections:\n\n{context}"
                )
            },
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        answer = response.choices[0].message.content
        st.session_state.chat_history.append(("assistant", answer))

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
