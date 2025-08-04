from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uvicorn

from utils import process_text
from models import get_embedding_model, get_classifier_model
from rag import create_vector_db, rag_query
from classification import classify_sentiment

app = FastAPI(title="Análise de Sentimento em Arquivos de Texto")

# Variáveis globais
embedding_model = get_embedding_model()
classifier_model, tokenizer = get_classifier_model()
vector_db = None

@app.post("/process_documents")
async def process_documents(files: List[UploadFile] = File(...)):
    global vector_db
    texts = []
    for file in files:
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Apenas arquivos .txt são permitidos.")
        content = await file.read()
        texts.append(process_text(content.decode('utf-8')))
    
    vector_db = create_vector_db(texts, embedding_model)
    return {"message": "Documentos processados com sucesso."}

@app.post("/rag_query")
async def perform_rag_query(query: str):
    if vector_db is None:
        raise HTTPException(status_code=400, detail="Processar documentos primeiro.")
    response = rag_query(query, vector_db, embedding_model)
    return {"response": response}

@app.post("/classify_sentiment")
async def perform_classification(text: str):
    sentiment, score = classify_sentiment(text, classifier_model, tokenizer)
    return {"sentiment": sentiment, "score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

