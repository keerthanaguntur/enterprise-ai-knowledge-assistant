from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

documents = []

# Load documents
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} document pages")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Create embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Build vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save locally
vectorstore.save_local(VECTORSTORE_DIR)

print("Vector store built and saved successfully")
