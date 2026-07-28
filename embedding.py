from langchain_huggingface import HuggingFaceEmbeddings

def get_emb():
    emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return emb