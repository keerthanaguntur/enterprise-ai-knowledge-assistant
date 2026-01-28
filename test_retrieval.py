from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"

# Use modern Ollama embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Load vector store (safe because YOU created it)
vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

query = "What is this document about?"

results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content[:300])
    print("Source:", doc.metadata)
