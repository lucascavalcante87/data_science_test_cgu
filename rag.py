import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
generator_model = GPT2LMHeadModel.from_pretrained('gpt2')

def create_vector_db(texts: list, embedding_model) -> faiss.IndexFlatL2:
    """Cria banco de vetores FAISS com embeddings dos textos."""
    embeddings = embedding_model.encode(texts)
    dimension = embeddings.shape[1]
    vector_db = faiss.IndexFlatL2(dimension)
    vector_db.add(np.array(embeddings))
    return vector_db

def rag_query(query: str, vector_db: faiss.IndexFlatL2, embedding_model) -> str:
    """Realiza consulta RAG: recupera contexto e gera resposta."""
    query_embedding = embedding_model.encode([query])
    _, indices = vector_db.search(np.array(query_embedding), k=3)

    contexts = ["Contexto recuperado " + str(i) for i in indices[0]]
    prompt = f"Query: {query}\nContext: {' '.join(contexts)}\nResponse:"
    inputs = generator_tokenizer.encode(prompt, return_tensors='pt')
    outputs = generator_model.generate(inputs, max_length=100)
    return generator_tokenizer.decode(outputs[0])
