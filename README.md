# Advanced RAG System with Multi-Format Support

A comprehensive Retrieval-Augmented Generation (RAG) system that supports multiple document formats including PDFs, Word documents, text files, CSVs, and images using OCR. Built with LangChain, Google Gemini, and QdrantDB.

## 🌟 Features

- **Multi-format Support**: Process PDFs, Word documents, text files, CSVs, and images (with OCR)
- **AI-Powered**: Leverages Google Gemini for advanced text and image understanding
- **Hybrid Text Extraction**: Combines OCR (Tesseract, EasyOCR) with AI-powered text extraction
- **Scalable Storage**: Uses Qdrant vector database for efficient similarity search
- **Command-line Interface**: Simple and intuitive CLI for all operations
- **Configurable**: Customize chunking, embedding models, and search parameters
- **Fallback Mechanisms**: Automatically falls back to alternative methods when primary methods fail

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Google API Key (for Gemini)
- QdrantDB (runs locally by default)
- Tesseract OCR (for fallback text extraction)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saichandra1199/Standard_RAG.git
   cd Standard_RAG
   ```

2. **Set up a virtual environment (recommended)**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

5. **Configure environment variables**:
   Create a `.env` file in the project root with your API key:
   ```env
   # Required
   GOOGLE_API_KEY=your_google_api_key_here
   
   # Optional (with defaults)
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=documents
   EMBEDDING_MODEL=text-embedding-3-small
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   MAX_TOKENS=1000
   TEMPERATURE=0.1
   ```

## 🛠️ Usage

### Command Line Interface

The system provides a simple CLI for all operations:

```
Usage: cli.py [command] [arguments]

Commands:
  add <files...>    Add documents to the knowledge base
  query <question>  Query the knowledge base
  clear            Clear the knowledge base
  status           Show status of the knowledge base
```

### Adding Documents

Add one or more documents to the knowledge base:

```bash
# Add multiple documents
python cli.py add document1.pdf document2.docx image.jpg

# Add all documents in a directory
python cli.py add ./documents/*
```

### Querying the Knowledge Base

Ask questions about your documents:

```bash
python cli.py query "What are the key findings in these documents?"
```

### Managing the Knowledge Base

```bash
# Clear all documents
python cli.py clear

# Check status
python cli.py status
```

## 🏗️ Architecture

The system is built with a modular architecture:

```
Standard_RAG/
├── document_processor.py  # Handles document loading and processing
├── vector_store.py       # Manages vector embeddings and storage
├── rag_system.py         # Core RAG implementation
├── config.py            # Configuration management
└── cli.py               # Command-line interface
```

## 🔍 How It Works

1. **Document Processing**:
   - Supports multiple document formats (PDF, DOCX, TXT, CSV, images)
   - Extracts text using appropriate methods for each format
   - For images and PDFs, uses Google Gemini Vision API with fallback to Tesseract/EasyOCR

2. **Text Chunking**:
   - Splits documents into manageable chunks
   - Maintains context with overlapping chunks
   - Configurable chunk size and overlap

3. **Embedding Generation**:
   - Converts text chunks into vector embeddings
   - Uses efficient embedding models for semantic search

4. **Vector Storage**:
   - Stores embeddings in Qdrant vector database
   - Enables fast similarity search

5. **Query Processing**:
   - Converts natural language queries to embeddings
   - Retrieves most relevant document chunks
   - Generates contextual responses using Gemini

## ⚙️ Configuration

Customize the system behavior by modifying the `.env` file:

- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `QDRANT_URL`: Qdrant database URL (default: http://localhost:6333)
- `EMBEDDING_MODEL`: Embedding model to use (default: text-embedding-3-small)
- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `MAX_TOKENS`: Maximum tokens for generated responses (default: 1000)
- `TEMPERATURE`: Response creativity (0.0 to 1.0, default: 0.1)

## 🤖 Supported Models

- **Text Generation**: Google Gemini 1.5 Pro
- **Embeddings**: text-embedding-3-small (default)
- **OCR**: Tesseract (primary), EasyOCR (fallback)

## 📝 Notes

- The system includes rate limiting and retry logic for API calls
- For large document collections, consider increasing the Qdrant resources
- Image processing requires additional dependencies (Pillow, OpenCV)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Gemini](https://ai.google.dev/)
- [Qdrant](https://qdrant.tech/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)

### Querying the Knowledge Base

```bash
python cli.py query "Your question here"
```

### Clearing the Knowledge Base

```bash
python cli.py clear
```

## Configuration

Edit the `.env` file to customize the following settings:

- `QDRANT_URL`: URL of your Qdrant instance (default: http://localhost:6333)
- `QDRANT_COLLECTION`: Collection name (default: documents)
- `EMBEDDING_MODEL`: OpenAI embedding model (default: text-embedding-3-small)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200)
- `MAX_TOKENS`: Maximum tokens for responses (default: 1000)
- `TEMPERATURE`: Response creativity (0.0 to 1.0, default: 0.1)

## Running Qdrant

### Local Installation

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Cloud Option

Sign up at [Qdrant Cloud](https://qdrant.tech/cloud/) for a managed Qdrant service.

## Example Workflow

1. Start Qdrant:
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

2. Add documents to the knowledge base:
   ```bash
   python cli.py add documents/*
   ```

3. Ask questions:
   ```bash
   python cli.py query "What are the key points from the documents?"
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
