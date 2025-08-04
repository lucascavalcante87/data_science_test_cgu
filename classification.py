from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def classify_sentiment(text: str, model, tokenizer) -> tuple:
    """Classifica o sentimento do texto."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label = 'positivo' if probs[0][1] > 0.5 else 'negativo'  # Adaptação simples para PT-BR
    score = probs[0][1].item()
    return label, score
