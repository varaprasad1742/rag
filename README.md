# Production RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI that enables intelligent question-answering over PDF documents using state-of-the-art NLP techniques.

## Overview

This RAG system provides a complete pipeline for ingesting PDF documents, processing them into semantic chunks, storing embeddings in a vector database, and answering questions based on the ingested content using Google's Gemini LLM.

## Features

- **📄 PDF Document Ingestion**: Upload and process multiple PDF files simultaneously
- **🔍 Semantic Search**: Fast vector similarity search using FAISS with HNSW indexing
- **🎯 Re-ranking**: Improve retrieval quality with cross-encoder re-ranking
- **🤖 AI-Powered Answers**: Generate contextual answers using Google Gemini
- **💾 Redis Caching**: Multi-layer caching for embeddings, retrieval, and generation
- **🐳 Docker Support**: Fully containerized deployment with Docker Compose
- **⚡ High Performance**: Optimized with batch processing and caching strategies
- **🔄 Thread-Safe**: Concurrent request handling with proper synchronization

## Architecture

The system follows a modular architecture with distinct layers:

```
┌─────────────┐
│   FastAPI   │  ← REST API Layer
├─────────────┤
│  Ingestion  │  ← PDF Loading → Chunking → Embedding → Vector Storage
├─────────────┤
│  Retrieval  │  ← Query Embedding → ANN Search → Re-ranking
├─────────────┤
│ Generation  │  ← Context Building → LLM Generation
├─────────────┤
│   Caching   │  ← Redis for Multi-layer Caching
└─────────────┘
```

### Pipeline Flow

1. **Ingestion Pipeline**:
   - Upload PDF files via API
   - Extract text from PDFs using PyPDF
   - Split text into overlapping chunks (500 words, 100 word overlap)
   - Generate embeddings using Sentence Transformers
   - Store vectors in FAISS HNSW index
   - Cache embeddings in Redis

2. **Query Pipeline**:
   - Embed user query
   - Perform approximate nearest neighbor (ANN) search (top-20)
   - Re-rank results using cross-encoder (top-5)
   - Build context from top chunks
   - Generate answer using Gemini LLM
   - Cache results at each stage

## Technology Stack

### Core Framework
- **FastAPI**: Modern, high-performance web framework
- **Uvicorn**: ASGI server for production deployment

### NLP & ML
- **Sentence Transformers**: Text embeddings (`all-MiniLM-L6-v2`)
- **FAISS**: Vector similarity search with HNSW algorithm
- **CrossEncoder**: Re-ranking model (`ms-marco-MiniLM-L-6-v2`)
- **Google Gemini**: Large language model for answer generation
- **PyTorch**: Deep learning backend

### Data Processing
- **PyPDF**: PDF text extraction
- **NumPy**: Numerical operations

### Infrastructure
- **Redis**: Caching layer for embeddings and results
- **Docker**: Containerization
- **Pydantic**: Data validation and settings management

## Installation

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Redis server (or use Docker Compose)

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/varaprasad1742/rag.git
cd rag
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
Create a `.env` file in the root directory:
```env
APP_NAME=Production RAG System
ENV=development

REDIS_HOST=localhost
REDIS_PORT=6379

LOG_LEVEL=INFO

GENAI_API_KEY=your_google_gemini_api_key_here
```

5. **Start Redis** (if not using Docker):
```bash
redis-server
```

6. **Run the application**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Setup

1. **Configure environment variables**:
Create a `.env` file as shown above.

2. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | Production RAG System |
| `ENV` | Environment (development/production) | development |
| `REDIS_HOST` | Redis server hostname | localhost |
| `REDIS_PORT` | Redis server port | 6379 |
| `LOG_LEVEL` | Logging level | INFO |
| `GENAI_API_KEY` | Google Gemini API key | Required |

### Model Configuration

You can customize the models by modifying the initialization parameters in the code:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Re-ranking Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM Model**: `gemini-3-flash-preview`

### Chunking Parameters

- **Chunk Size**: 500 words
- **Overlap**: 100 words

### Retrieval Parameters

- **Top-K ANN Results**: 20
- **Top-N Re-ranked Results**: 5
- **Max Context Characters**: 8000

## API Documentation

### Health Check

**GET** `/health`

Check if the service is running.

**Response**:
```json
{
  "status": "ok"
}
```

### Ingest PDFs

**POST** `/ingest/pdfs`

Upload and process PDF files for indexing.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `files` (array of PDF files)

**Response**:
```json
{
  "total_files": 2,
  "results": [
    {
      "file": "document1.pdf",
      "status": "ingested",
      "num_chunks": 45
    },
    {
      "file": "document2.pdf",
      "status": "ingested",
      "num_chunks": 32
    }
  ]
}
```

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/ingest/pdfs" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### Query

**POST** `/query/`

Ask questions about the ingested documents.

**Request**:
```json
{
  "query": "What is the main topic discussed in the documents?"
}
```

**Response**:
```json
{
  "query": "What is the main topic discussed in the documents?",
  "answer": "The documents discuss...",
  "sources": [
    {
      "file": "document1.pdf",
      "chunk_id": "a3f5d8e..."
    },
    {
      "file": "document2.pdf",
      "chunk_id": "b7c9a1f..."
    }
  ]
}
```

**Example using cURL**:
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

### Interactive API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
rag/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── ingest.py          # PDF ingestion endpoints
│   │   └── query.py           # Query endpoints
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging setup
│   │   └── cache.py           # Redis client
│   ├── ingestion/
│   │   ├── pdf_loader.py      # PDF text extraction
│   │   ├── chunker.py         # Text chunking logic
│   │   └── embedder.py        # Embedding generation with caching
│   ├── retrieval/
│   │   ├── hnsw.py            # FAISS HNSW index management
│   │   ├── retriever.py       # ANN search with caching
│   │   ├── reranker.py        # Cross-encoder re-ranking
│   │   └── utils.py           # Utility functions
│   └── generation/
│       └── llm.py             # Gemini LLM integration
├── data/                       # Data storage (FAISS index, metadata)
├── docker/
│   └── Dockerfile             # Container configuration
├── docker-compose.yml         # Multi-container setup
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Performance Optimization

The system implements several optimization strategies:

1. **Multi-layer Caching**:
   - Embeddings cached indefinitely
   - ANN results cached for 5 minutes
   - Re-ranking results cached for 5 minutes
   - Final answers cached for 10 minutes

2. **Batch Processing**:
   - Embeddings computed in batches (batch_size=32)
   - Normalized embeddings for faster similarity computation

3. **Efficient Indexing**:
   - HNSW (Hierarchical Navigable Small World) for fast ANN search
   - Thread-safe index updates

4. **Vector Storage**:
   - Persistent FAISS index on disk
   - Metadata stored in JSONL format

## Usage Examples

### Example 1: Ingest Documents

```python
import requests

url = "http://localhost:8000/ingest/pdfs"
files = [
    ("files", open("research_paper.pdf", "rb")),
    ("files", open("technical_doc.pdf", "rb"))
]

response = requests.post(url, files=files)
print(response.json())
```

### Example 2: Query the System

```python
import requests

url = "http://localhost:8000/query/"
payload = {
    "query": "What are the key findings of the research?"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## Development

### Running Tests

```bash
# Add your test commands here
pytest tests/
```

### Code Style

The project follows Python best practices:
- Type hints for better code clarity
- Modular architecture for maintainability
- Comprehensive error handling

### Adding New Features

1. Create a new module in the appropriate directory
2. Update the API routers in `app/api/`
3. Add configuration options to `app/core/config.py`
4. Update this README with documentation

## Troubleshooting

### Common Issues

1. **Redis Connection Error**:
   - Ensure Redis is running: `redis-cli ping`
   - Check `REDIS_HOST` and `REDIS_PORT` in `.env`

2. **FAISS Index Not Found**:
   - The index is created automatically on first use
   - Ensure the `data/` directory is writable

3. **Out of Memory**:
   - Reduce batch size in embedding generation
   - Limit the number of concurrent requests

4. **Slow Query Performance**:
   - Check Redis cache hit rate
   - Verify FAISS index is loaded correctly
   - Consider adjusting `ef_search` parameter

### Logs

View application logs:
```bash
# Docker
docker-compose logs -f api

# Local
# Logs are printed to stdout
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please add an appropriate license file.

## Acknowledgments

- **Sentence Transformers**: Efficient text embeddings
- **FAISS**: Fast similarity search library by Meta AI
- **FastAPI**: Modern Python web framework
- **Google Gemini**: Advanced language model

## Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ using modern NLP and MLOps practices.
