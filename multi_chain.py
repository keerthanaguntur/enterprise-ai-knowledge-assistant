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

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = OllamaLLM(model="llama3")

query = "What is this document about?"

# Retrieve context
docs = retriever.invoke(query)
context = "\n\n".join(doc.page_content for doc in docs)

# --- CHAIN 1: SUMMARY ---
summary_prompt = f"""
You are an enterprise analyst.
Summarize the following content in 5 bullet points.

Content:
{context}

Summary:
"""

summary = llm.invoke(summary_prompt)

# --- CHAIN 2: ACTION ITEMS ---
action_prompt = f"""
You are a business consultant.
Extract clear action items from the content below.

Content:
{context}

Action Items:
"""

actions = llm.invoke(action_prompt)

# --- CHAIN 3: RISKS / DECISIONS ---
risk_prompt = f"""
You are a risk analyst.
Identify potential risks, decisions, or open questions in the content.

Content:
{context}

Risks / Decisions:
"""

risks = llm.invoke(risk_prompt)

print("\n=== SUMMARY ===")
print(summary)

print("\n=== ACTION ITEMS ===")
print(actions)

print("\n=== RISKS / DECISIONS ===")
print(risks)
