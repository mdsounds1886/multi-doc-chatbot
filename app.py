import os
import streamlit as st
import pdfplumber
import docx
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Initialize OpenAI client (v1.x syntax)
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

# Load and embed all document content
@st.cache_resource
def load_documents():
    chunks = []
    docs_folder = "./docs"
    files = [f for f in os.listdir(docs_folder) if f.lower().endswith(('.pdf', '.docx', '.xlsx', '.xls', '.txt'))]
    
    for file in files:
        path = os.path.join(docs_folder, file)
        raw_text = extract_text_from_file(path)
        if raw_text:
            chunks.extend(raw_text.split("\n\n"))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return chunks, index, model

# Load documents once
chunks, index, model = load_documents()

# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask a question about the documents")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # Embed the question and retrieve similar chunks
    query_embedding = model.encode([user_input])
    _, indices = index.search(np.array(query_embedding), k=3)
    context = "\n\n".join([chunks[i] for i in indices[0]])

    # OpenAI chat completion
    messages = [
    {"role": "system", "content": f"You’re a sharp, slightly sassy assistant who gives clear, business-casual answers with a touch of dry humor. Don’t ramble — be efficient, maybe crack a subtle joke, and stick strictly to the info below:\n\n{context}"},
    {"role": "user", "content": user_input}
]

    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    answer = response.choices[0].message.content
    st.session_state.chat_history.append(("assistant", answer))

# Display conversation
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

