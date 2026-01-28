from langchain_community.document_loaders import PyPDFLoader
import os

DATA_DIR = "data"

all_documents = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        documents = loader.load()
        all_documents.extend(documents)

print(f"Loaded {len(all_documents)} document pages")

# Inspect one document
print("\n--- Sample Document ---")
print(all_documents[0].page_content[:500])
print("\nMetadata:", all_documents[0].metadata)
