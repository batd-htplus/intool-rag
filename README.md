# RAG Platform Internal Knowledge System

## 📋 Overview

A production-ready RAG (Retrieval-Augmented Generation) system for internal knowledge management. Supports multilingual chat (Vietnamese, English, Japanese) with document ingestion and AI-powered question answering.

**Key Features:**
- ✅ OpenAI-compatible API (works with Open WebUI)
- ✅ Multi-language support (EN/VI/JA)
- ✅ Multiple document formats (PDF, DOCX, XLSX)
- ✅ 100% Open-source & self-hosted
- ✅ Production-ready architecture
- ✅ No cloud vendor lock-in

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface                        │
│              (Open WebUI - Free OSS)                    │
└─────────────────┬───────────────────────────────────────┘
                  │
        OpenAI-compatible API (/v1/chat/completions)
                  │
┌─────────────────▼───────────────────────────────────────┐
│         Backend API (FastAPI)                           │
│   • Auth & Permissions                                  │
│   • Document Management                                 │
│   • Request/Response Mapping                            │
└─────────────────┬───────────────────────────────────────┘
                  │
            HTTP Internal API
                  │
┌─────────────────▼───────────────────────────────────────┐
│        RAG Service (AI Plane)                           │
│   • Query Processing                                    │
│   • Document Retrieval                                  │
│   • LLM Integration                                     │
└──┬──────────────────────────────┬──────────┬────────────┘
   │                              │          │
┌──▼───────┐  ┌──────────────┐  ┌▼─────┐  ┌▼──────────┐
│ Qdrant   │  │  BGE-M3      │  │Qwen  │  │  Pipeline │
│Vector DB │  │ Embeddings   │  │ LLM  │  │  Chunker  │
└──────────┘  └──────────────┘  └──────┘  └───────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- (Optional) NVIDIA GPU with CUDA for faster processing

### 1. Clone & Setup
```bash
cd /home/administrator/Workspace/Tools/in-rag-tools
```

### 2. Configure (Optional)
```bash
cp .env.example .env
# Edit .env if needed
```

### 3. Start Services
```bash
docker-compose up -d
```

### 4. Verify Services
```bash
# Backend health
curl http://localhost:8000/health

# RAG service health
curl http://localhost:8001/health

# Qdrant ready
curl http://localhost:6333/health
```

### 5. Access Web UI
- **Open WebUI**: http://localhost:3000
- **Backend API**: http://localhost:8000/docs
- **RAG Service**: http://localhost:8001/docs

---

## 📚 API Usage

### Upload Document
```bash
curl -X POST http://localhost:8000/v1/ingest/upload \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "project=my-project" \
  -F "language=en"
```

### Chat with Documents (OpenAI-compatible)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "What is in the documents?"}
    ],
    "temperature": 0.7,
    "project": "my-project"
  }'
```

### Streaming Chat
```bash
curl -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "qwen",
    "messages": [
      {"role": "user", "content": "Explain the documents"}
    ],
    "stream": true
  }'
```

### List Documents
```bash
curl http://localhost:8000/v1/documents?project=my-project
```

---

## 🔧 Configuration

### Backend (`backend/app/core/config.py`)
- `RAG_SERVICE_URL`: RAG service endpoint
- `AUTH_ENABLED`: Enable authentication
- `LOG_LEVEL`: Logging level (INFO/DEBUG/ERROR)

### RAG (`rag/config.py`)
- `EMBEDDING_MODEL`: BGE-M3 multi-language embeddings
- `LLM_MODEL`: Qwen2.5 7B instruct model
- `EMBEDDING_DEVICE`: "cpu" or "cuda"
- `LLM_DEVICE`: "cpu" or "cuda"
- `CHUNK_SIZE`: Document chunk size (512)
- `RETRIEVAL_TOP_K`: Number of chunks to retrieve (5)

### Environment Variables
Create `.env` file:
```env
RAG_SERVICE_URL=http://rag-service:8001
QDRANT_URL=http://qdrant:6333
EMBEDDING_DEVICE=cpu
LLM_DEVICE=cpu
AUTH_ENABLED=false
LOG_LEVEL=INFO
DEBUG=false
```

---

## 📊 Supported Document Formats

| Format | Extension | Support |
|--------|-----------|---------|
| PDF    | .pdf      | ✅      |
| Word   | .docx     | ✅      |
| Excel  | .xlsx     | ✅      |
| TXT    | .txt      | ⏳      |

---

## 🧠 LLM & Embedding Models

### Embedding: BGE-M3
- **Multi-language**: English, Chinese, Vietnamese, Japanese, etc.
- **Dimension**: 1024
- **Size**: ~1GB
- **License**: Open-source
- **Source**: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

### LLM: Qwen2.5
- **Language**: Primarily English, supports other languages
- **Size**: 7B parameters
- **Memory**: ~16GB VRAM (or use CPU)
- **License**: Open-source (Alibaba)
- **Source**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

### Vector DB: Qdrant
- **Type**: Vector database
- **License**: Open-source
- **Memory**: Configurable
- **API**: REST + gRPC

---

## 🔐 Security

### Current Setup (Development)
- No authentication enabled by default
- Open CORS for development

### Production Recommendations
1. Enable `AUTH_ENABLED` in config
2. Implement JWT token validation
3. Use API keys for external access
4. Set strong Qdrant API key
5. Use HTTPS reverse proxy (nginx/traefik)
6. Restrict CORS to specific domains
7. Add rate limiting
8. Use environment-based secrets

---

## 📈 Performance Tuning

### GPU Acceleration (Recommended)
```bash
# In docker-compose.yml or .env:
EMBEDDING_DEVICE=cuda
LLM_DEVICE=cuda

# Add GPU support to docker-compose:
services:
  rag-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Memory Optimization
- Adjust `CHUNK_SIZE` for longer/shorter chunks
- Reduce `RETRIEVAL_TOP_K` to retrieve fewer documents
- Use smaller embedding/LLM models if needed

### Batch Processing
- Documents are processed asynchronously
- Monitor `/v1/ingest/status/{doc_id}` for progress

---

## 🐛 Troubleshooting

### Service won't start
```bash
docker-compose logs backend
docker-compose logs rag-service
docker-compose logs qdrant
```

### Out of memory
```bash
# Reduce LLM model or use CPU
EMBEDDING_DEVICE=cpu
LLM_DEVICE=cpu
```

### Slow responses
- Add GPU support
- Increase `RETRIEVAL_TOP_K` for better results
- Check document quality

---

## 📝 Project Structure

```
.
├── backend/                  # FastAPI Backend
│   ├── app/
│   │   ├── api/             # API endpoints
│   │   ├── core/            # Config, auth, logging
│   │   ├── domain/          # Models, permissions
│   │   ├── schemas/         # Request/response schemas
│   │   ├── services/        # Business logic
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── rag/                      # RAG Service
│   ├── embedding/           # BGE-M3 embeddings
│   ├── ingest/              # Document loading & chunking
│   ├── llm/                 # Qwen LLM
│   ├── query/               # Retrieval & prompting
│   ├── vector_store/        # Qdrant interface
│   ├── config.py
│   └── index.py             # Query engine
├── docker-compose.yml       # Service orchestration
└── README.md
```

---

## 🧪 Testing

### Health Checks
```bash
# All services healthy
docker-compose ps

# Individual health checks
docker exec rag-backend curl http://localhost:8000/health
docker exec rag-service curl http://localhost:8001/health
docker exec qdrant curl http://localhost:6333/health
```

### Sample Query
```bash
# 1. Upload document first
curl -X POST http://localhost:8000/v1/ingest/upload \
  -F "file=@sample.pdf" \
  -F "project=test"

# 2. Query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "What is in this document?"}],
    "project": "test"
  }'
```

---

## 📦 Deployment

### Docker Compose (Development/Small Scale)
```bash
docker-compose up -d
```

### Kubernetes (Production)
See `k8s/` directory for Helm charts and manifests.

### Hardware Requirements

**Minimum (CPU only):**
- CPU: 4+ cores
- RAM: 16GB
- Storage: 20GB

**Recommended (GPU):**
- CPU: 8+ cores
- GPU: NVIDIA A100 / RTX 4090 (24GB+)
- RAM: 32GB
- Storage: 50GB

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Submit pull request
4. Follow coding standards

---

## 📄 License

MIT License - See LICENSE file

---

## 🆘 Support

- **Documentation**: See `/docs` folder
- **Issues**: GitHub Issues
- **Email**: support@example.com

---

## 🎯 Roadmap

- [ ] Multi-turn conversation memory
- [ ] Fine-tuning capability
- [ ] Advanced filtering & permissions
- [ ] Web crawler integration
- [ ] Real-time document sync
- [ ] Analytics dashboard
- [ ] Multi-LLM support
- [ ] Function calling (tool use)

---

**Last Updated**: 2026-01-12  
**Version**: 0.1.0
