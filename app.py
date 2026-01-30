import streamlit as st
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Enterprise AI Knowledge Assistant", layout="wide")

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

st.title("💼 Enterprise AI Knowledge Assistant")
st.caption("LangChain + Ollama | RAG + Multi-Chain Intelligence")

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(DATA_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success("Files uploaded successfully")

    # Rebuild vector store
    documents = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    st.sidebar.success("Knowledge base updated")

# --- Main Area ---
query = st.text_input("Ask a question about your documents")

if query and os.path.exists(VECTORSTORE_DIR):
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = OllamaLLM(model="llama3")

    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Answer
    answer_prompt = f"""
You are an enterprise AI assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}

Answer:
"""
    answer = llm.invoke(answer_prompt)

    # Summary
    summary = llm.invoke(
        f"Summarize the following content in 5 bullet points:\n{context}"
    )

    # Actions
    actions = llm.invoke(
        f"Extract clear action items from the content:\n{context}"
    )

    # Risks
    risks = llm.invoke(
        f"Identify risks or decisions in the content:\n{context}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💡 Answer")
        st.write(answer)

        st.subheader("📌 Summary")
        st.write(summary)

    with col2:
        st.subheader("✅ Action Items")
        st.write(actions)

        st.subheader("⚠️ Risks / Decisions")
        st.write(risks)

elif query:
    st.warning("Please upload documents first.")
