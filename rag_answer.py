from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

VECTORSTORE_DIR = "vectorstore"

# Load embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")

vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM
llm = OllamaLLM(model="llama3")

# Query
query = "What is this document about?"

# 1️⃣ Retrieve relevant documents
docs = retriever.invoke(query)

# 2️⃣ Build context manually
context = "\n\n".join(doc.page_content for doc in docs)

# 3️⃣ Grounded prompt
prompt = f"""
You are an enterprise AI assistant.
Answer ONLY using the context below.
If the answer is not present, say:
"I don't know based on the document."

Context:
{context}

Question:
{query}

Answer:
"""

# 4️⃣ Generate answer
answer = llm.invoke(prompt)

print("\n--- Answer ---")
print(answer)

print("\n--- Sources ---")
for doc in docs:
    print(doc.metadata)

