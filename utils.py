import re

def process_text(text: str) -> str:
    """Processa o texto: remove caracteres especiais, normaliza, etc."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.lower() 
    return text
