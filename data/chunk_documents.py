from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

DATA_DIR = "data"

documents = []

# Load PDFs
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} document pages")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")

# Inspect one chunk
print("\n--- Sample Chunk ---")
print(chunks[0].page_content[:500])
print("\nMetadata:", chunks[0].metadata)
