from langchain.embeddings import HuggingFaceEmbeddings

def get_hf_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
