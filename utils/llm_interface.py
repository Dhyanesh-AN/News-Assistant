from langchain_community.llms import Ollama

def get_local_llm(model_name="llama3"):
    return Ollama(model=model_name)
