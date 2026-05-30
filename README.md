# RAG Sample

Exemplo simples de Retrieval-Augmented Generation (RAG) com Python.

## O que é

Implementação básica de RAG que:
- Gera embeddings para documentos usando `text-embedding-3-small`
- Busca os top-k documentos mais similares via similaridade cosseno
- Gera respostas usando LLM com contexto recuperado

## Requisitos

- Python 3.13+
- uv (gerenciador de pacotes)
- Chave de API OpenAI

## Configuração

1. Copie o arquivo de exemplo de variáveis de ambiente:
   ```bash
   cp env.example .env
   ```

2. Configure as variáveis no `.env`:
   ```
   OPENAI_API_KEY=sua-chave-aqui
   OPENAI_BASE_URL=https://api.openai.com/v1
   AI_MODEL=google/gemma-4-31b-it:free
   ```

   - `AI_MODEL` (opcional): modelo de chat a ser usado. Padrão: `google/gemma-4-31b-it:free`

## Instalação e Execução

```bash
uv sync
uv run main.py
```

## Estrutura

```
rag-sample/
├── main.py          # Script principal
├── simplerag/
│   ├── __init__.py
│   └── rag.py       # Implementação do RAG
├── pyproject.toml
└── env.example
```