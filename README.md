# Análise de Sentimento em Arquivos de Texto

Este é um protótipo de API para análise de sentimento em arquivos de texto, utilizando técnicas de embeddings, RAG (Retrieval-Augmented Generation) e classificação.

## Configuração e Execução

1. Clone o repositório:
   ```
   git clone https://github.com/lucascavalcante87/text_files_sentiment_analysis.git
   cd text_files_sentiment_analysis
   ```

2. Crie um ambiente virtual (opcional, mas recomendado):
   ```
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

4. Execute a aplicação:
   ```
   python app.py
   ```

A API estará disponível em `http://localhost:8000`. Use ferramentas como Postman ou curl para testar os endpoints.

## Endpoints

- **POST /process_documents**: Processa arquivos de texto enviados e armazena embeddings.
  - Corpo: Multipart form com arquivos de texto.
  - Resposta: Confirmação de processamento.

- **POST /rag_query**: Realiza consulta RAG em documentos processados.
  - Corpo: JSON com "query" (string).
  - Resposta: Resposta gerada com contexto recuperado.

- **POST /classify_sentiment**: Classifica o sentimento de um texto.
  - Corpo: JSON com "text" (string).
  - Resposta: Sentimento (positivo, negativo, neutro) e pontuação.

## Arquitetura

Para uma visão da arquitetura do sistema (focada na versão de produção com microservices para isolamento), consulte o arquivo [architecture_design.txt](architecture_design.txt). Ele descreve como o sistema pode evoluir para escalabilidade, com componentes isolados.

## Dependências Principais

- FastAPI: Para a API web.
- Sentence Transformers: Para embeddings.
- FAISS: Para banco de vetores.
- Transformers (Hugging Face): Para modelos de geração e classificação.

O código está modularizado em arquivos separados para facilitar a manutenção e evolução para produção.

## Licença

MIT License.
```
