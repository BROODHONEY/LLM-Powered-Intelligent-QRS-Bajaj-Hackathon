# LLM-Powered Intelligent Q&A Service - Bajaj Hackathon

A sophisticated, high-performance Question-Answering system built with FastAPI that leverages Large Language Models (LLMs) and advanced retrieval techniques to provide accurate, context-aware answers from document collections.

## 🚀 Features

### Core Capabilities
- **Document Ingestion**: Automatically processes and indexes documents from URLs
- **Intelligent Chunking**: Advanced text segmentation with configurable overlap
- **Vector Search**: FAISS-based semantic search for relevant content retrieval
- **LLM Integration**: HuggingFace Transformers integration for answer generation
- **Smart Context Building**: Multi-level fallback strategies for comprehensive context
- **Advanced Reranking**: Custom scoring algorithms considering semantic similarity, keyword overlap, and numeric presence

### Performance Optimizations
- **Caching System**: Intelligent document caching to avoid redundant processing
- **Embedding Reuse**: Single embedder instance for improved performance
- **Configurable Parameters**: Environment-based tuning for different use cases
- **Efficient Retrieval**: Optimized chunk selection and context building

### Answer Quality
- **Comprehensive Coverage**: Multi-chunk analysis for complete answers
- **Precise Information**: Exact quoting of numbers, dates, and conditions
- **Context Awareness**: Answers grounded in provided documents only
- **Consistent Formatting**: Clean, concise responses without citations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App  │───▶│  Document Input  │───▶│  Text Chunking  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Query Input   │    │   Embedding      │    │  FAISS Index   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Semantic      │    │  Context         │    │  LLM Answer    │
│  Search        │───▶│  Building        │───▶│  Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python 3.8+)
- **LLM**: HuggingFace Transformers (Qwen2.5-3B-Instruct)
- **Embeddings**: SentenceTransformers (intfloat/e5-base)
- **Vector Database**: FAISS
- **Text Processing**: pdfplumber, regex
- **Dependencies**: PyTorch, NumPy, pandas

## 📋 Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU support for faster LLM inference (optional)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd LLM-Powered-Intelligent-QRS-Bajaj-Hackathon
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
# Model Configuration
EMBEDDER_MODEL=intfloat/e5-base
HF_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct

# Chunking Parameters
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# Retrieval Parameters
TOP_K=12
CANDIDATE_MULTIPLIER=20
MAX_CONTEXT_TOKENS=2500

# Generation Parameters
MAX_NEW_TOKENS=128

# API Configuration
API_KEY=your_api_key_here
```

## 🎯 Usage

### 1. Start the Server
```bash
python app.py
```

The server will start on `http://localhost:8000`

### 2. API Endpoint

#### POST `/hackrx/run`

**Request Body:**
```json
{
  "documents": ["https://example.com/document.pdf"],
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The grace period for premium payment is 30 days.",
    "The waiting period for pre-existing diseases is 36 months."
  ]
}
```

### 3. Example Usage with cURL
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d '{
    "documents": ["https://example.com/policy.pdf"],
    "questions": ["What is covered under this policy?"]
  }'
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Size of text chunks in characters |
| `CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `TOP_K` | 12 | Number of top chunks to retrieve |
| `CANDIDATE_MULTIPLIER` | 20 | Multiplier for FAISS candidates |
| `MAX_CONTEXT_TOKENS` | 2500 | Maximum context length for LLM |
| `MAX_NEW_TOKENS` | 128 | Maximum tokens for answer generation |

### Performance Tuning

- **For Speed**: Decrease `TOP_K`, `CANDIDATE_MULTIPLIER`, `MAX_CONTEXT_TOKENS`
- **For Accuracy**: Increase `TOP_K`, `CANDIDATE_MULTIPLIER`, `MAX_CONTEXT_TOKENS`
- **For Memory**: Decrease `CHUNK_SIZE`, `MAX_CONTEXT_TOKENS`

## 🔧 Advanced Features

### Custom Embedding Models
The system supports various embedding models:
- **E5 Models**: Automatically handles "query:" and "passage:" prefixes
- **Instructor Models**: Task-specific embeddings
- **Sentence Transformers**: General-purpose embeddings

### Smart Context Building
1. **Sentence Selection**: Prioritizes most relevant sentences
2. **Chunk Expansion**: Includes additional sentences when needed
3. **Fallback Strategies**: Multiple levels of context expansion
4. **Length Optimization**: Balances coverage with token limits

### Intelligent Reranking
- **Semantic Similarity**: FAISS scores
- **Keyword Overlap**: Question-chunk word matching
- **Numeric Presence**: Bonus for chunks with numbers/dates
- **Must-have Terms**: Prioritizes chunks with essential keywords

## 📊 Performance Metrics

### Typical Response Times
- **Document Ingestion**: 30-60 seconds (first time)
- **Query Processing**: 15-30 seconds
- **Context Building**: 2-5 seconds
- **Answer Generation**: 10-20 seconds

### Memory Usage
- **Base Memory**: ~2GB
- **With Index**: +1-2GB (depending on document size)
- **Peak Memory**: +2-4GB during processing

## 🐛 Troubleshooting

### Common Issues

1. **"I don't know" Responses**
   - Check if documents contain relevant information
   - Increase `TOP_K` and `CANDIDATE_MULTIPLIER`
   - Verify chunking parameters

2. **Slow Performance**
   - Reduce `TOP_K` and `MAX_CONTEXT_TOKENS`
   - Use smaller embedding models
   - Enable GPU acceleration

3. **Memory Errors**
   - Decrease `CHUNK_SIZE` and `MAX_CONTEXT_TOKENS`
   - Process smaller documents
   - Increase system RAM

4. **Import Errors**
   - Ensure virtual environment is activated
   - Check Python version compatibility
   - Reinstall requirements

### Debug Mode
Enable detailed logging by setting environment variables:
```env
DEBUG=true
LOG_LEVEL=DEBUG
```

## 🔒 Security

- **API Key Authentication**: Required for all endpoints
- **Input Validation**: Sanitized document URLs and questions
- **Rate Limiting**: Configurable request limits
- **Error Handling**: Secure error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bajaj Hackathon** for the project opportunity
- **HuggingFace** for the transformer models
- **FAISS** for efficient vector search
- **FastAPI** for the robust web framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

---

**Built with ❤️ for intelligent document understanding and question answering**
