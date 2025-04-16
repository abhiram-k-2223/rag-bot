# Lightweight RAG API

A lightweight Retrieval-Augmented Generation (RAG) API using FAISS for vector storage and MiniLM for embeddings. This implementation is optimized for free-tier deployments.

## Features

- Fast and efficient vector search using FAISS
- Lightweight embeddings using MiniLM-L6-v2
- Simple text chunking and processing
- REST API interface using FastAPI
- Memory-efficient implementation

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your text file in the project directory
2. Start the API server:
```bash
python app.py
```

3. The API will be available at `http://localhost:8000`

### API Endpoints

#### POST /query
Query the RAG system with your text.

Request body:
```json
{
    "text": "Your query text here",
    "num_results": 3  // optional, defaults to 3
}
```

Response:
```json
{
    "results": ["relevant text chunk 1", "relevant text chunk 2", "relevant text chunk 3"],
    "scores": [0.95, 0.85, 0.75]  // similarity scores
}
```

## Deployment

This API is designed to be deployed on free-tier platforms. Some recommended platforms:
- Render
- Railway
- Heroku (free tier)
- Python Anywhere

## Memory Usage

- MiniLM-L6-v2 model: ~80MB
- FAISS index: Depends on your text size
- Base memory usage: ~100-200MB

## Limitations

- Maximum text file size: Depends on your deployment platform
- Chunk size: Default 1000 characters, adjustable in code
- Number of results: Limited by available memory 