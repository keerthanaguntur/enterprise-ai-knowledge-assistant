from langchain_ollama import OllamaLLM

# Initialize the local LLM via Ollama (new API)
llm = OllamaLLM(model="llama3")

prompt = "Explain what LangChain is and why companies use it."

response = llm.invoke(prompt)

print(response)
