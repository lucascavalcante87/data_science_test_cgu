# CGU Data Scientist Test Solution

## Overview
This repository contains a FastAPI-based API implementing the required endpoints for document processing, naive-RAG, and text classification. The solution is designed as a prototype that works locally but includes architectural considerations for scaling to 10,000 concurrent users.

## Technical Choices and Justifications

- **Embedding Model**: all-MiniLM-L6-v2 from sentence-transformers. Chosen for its efficiency, small size (fast inference), and good performance on semantic similarity tasks. Trade-off: Not as accurate as larger models like MPNet, but suitable for a prototype to reduce latency and resource usage.

- **Vector Database**: FAISS (in-memory for demo). Justified for its speed in similarity search and ease of integration. For production, recommend switching to Pinecone or Weaviate for managed, scalable vector storage. Trade-off: In-memory lacks persistence; production would use a persistent backend.

- **LLM/SLM for Generation and Classification**:
  - **Generation (RAG)**: GPT-2 (small model from HuggingFace). Chosen for local execution without API keys, fast loading. Trade-off: Responses may not be as coherent as larger models (e.g., Llama-2); in prod, use Grok API or similar for better quality.
  - **Classification**: DistilBERT fine-tuned on SST-2 for sentiment analysis. Efficient SLM, provides logits for logprobs. Justified for speed and accuracy on binary sentiment. Trade-off: Binary (POSITIVE/NEGATIVE); for multi-class, fine-tune further.

- **Chunking**: Custom overlap-based chunking. Configurable for flexibility. Trade-off: Simple character-based; could use semantic chunking (e.g., via langchain) for better quality but adds complexity.

- **Reranking**: BM25 for keyword-based rerank. Complements semantic search by handling lexical matches. Optional to allow trade-off between speed (no rerank) and precision.

- **Other**: PyPDF2 for PDF extraction (reliable, lightweight). Transformers for models (standard ecosystem).

## Challenges

- Handling large PDFs: Limited to memory; prod would use streaming.
- Concurrency: FastAPI is async, but models are CPU-bound; use GPU acceleration.
- Logprobs: Implemented via softmax on logits for transparency in classification decisions.

## Setup and Running

1. Install dependencies: `pip install -r requirements.txt`

2. Run the API: `python app.py`

3. Test endpoints using Postman or curl:
   - `/process_documents`: POST with files (multipart/form-data), query params chunk_size, chunk_overlap.
   - `/rag`: POST with JSON `{"question": "...", "use_rerank": true/false}`
   - `/classify`: POST with JSON `{"text": "..."}`

4. For testing, use `/reset_index` to clear the index.
