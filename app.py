# app.py
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import uuid
from typing import List, Dict
import logging

app = FastAPI(title="CGU Data Scientist Test API")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory FAISS index (for demo; in prod, use persistent like Pinecone)
dimension = 384  # For all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
metadata_store: List[Dict] = []  # Store metadata: {'file_name': str, 'chunk_id': int, 'text': str}

# Models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='gpt2')
sentiment_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline('sentiment-analysis', model=sentiment_model_name)

# Helper functions
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Invalid PDF file")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

class ProcessDocumentsResponse(BaseModel):
    message: str
    num_chunks: int

@app.post("/process_documents", response_model=ProcessDocumentsResponse)
async def process_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Query(500, description="Chunk size for text splitting"),
    chunk_overlap: int = Query(50, description="Overlap between chunks")
):
    global index, metadata_store
    all_embeddings = []
    all_metadata = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
        
        text = extract_text_from_pdf(file)
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk)
            all_embeddings.append(embedding)
            all_metadata.append({
                'file_name': file.filename,
                'chunk_id': i,
                'text': chunk
            })
    
    if all_embeddings:
        embeddings_np = np.array(all_embeddings).astype('float32')
        index.add(embeddings_np)
        metadata_store.extend(all_metadata)
    
    return {"message": "Documents processed and indexed successfully", "num_chunks": len(all_metadata)}

class RAGRequest(BaseModel):
    question: str
    use_rerank: bool = False

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict]

@app.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    global index, metadata_store
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet")
    
    question_embedding = embedder.encode(request.question).astype('float32')
    D, I = index.search(np.array([question_embedding]), k=5)  # Top 5
    
    retrieved_chunks = [metadata_store[i] for i in I[0] if i < len(metadata_store)]
    retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
    
    if request.use_rerank:
        bm25 = BM25Okapi([text.split() for text in retrieved_texts])
        scores = bm25.get_scores(request.question.split())
        sorted_indices = np.argsort(scores)[::-1]
        retrieved_chunks = [retrieved_chunks[idx] for idx in sorted_indices][:3]  # Top 3 after rerank
        retrieved_texts = [chunk['text'] for chunk in retrieved_chunks]
    
    context = "\n\n".join(retrieved_texts)
    prompt = f"Question: {request.question}\nContext: {context}\nAnswer:"
    
    try:
        generated = generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        answer = generated.split("Answer:")[-1].strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")
    
    sources = [{'file_name': chunk['file_name'], 'chunk_id': chunk['chunk_id']} for chunk in retrieved_chunks]
    
    return {"answer": answer, "sources": sources}

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    label: str
    score: float
    logprobs: Dict[str, float]

@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    inputs = sentiment_tokenizer(request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    label_id = torch.argmax(probs).item()
    label = sentiment_model.config.id2label[label_id]
    score = probs[0][label_id].item()
    
    logprobs = {sentiment_model.config.id2label[i]: torch.log(probs[0][i]).item() for i in range(probs.shape[1])}
    
    return {"label": label, "score": score, "logprobs": logprobs}

# Reset index for testing
@app.post("/reset_index")
async def reset_index():
    global index, metadata_store
    index = faiss.IndexFlatL2(dimension)
    metadata_store = []
    return {"message": "Index reset successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)