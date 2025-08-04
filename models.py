from sentence_transformers import SentenceTransformer
from transformers import pipeline

def get_embedding_model():
    """Carrega o modelo de embeddings."""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_classifier_model():
    """Carrega o modelo de classificação de sentimento."""
    classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return classifier.model, classifier.tokenizer  # Retorna model e tokenizer separadamente se necessário
