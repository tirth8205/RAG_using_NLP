# Local RAG Pipeline using NLP for PDF Q&A

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Natural Language Processing (NLP)** techniques to answer questions based on a user-specified PDF document. It runs locally, processing text, generating embeddings, retrieving relevant chunks, and producing answers with a user-selected Large Language Model (LLM) from Hugging Face.

## Features
- **Custom PDF Support**: Users can upload any PDF via a command-line argument or web UI.
- **Hugging Face Model Selection**: Choose any Hugging Face model (e.g., `google/gemma-2b-it`) for answer generation.
- **Local Execution**: Runs on your machine (CPU/GPU) for privacy and speed.
- **Modern Web UI**: Built with FastAPI backend and custom HTML/CSS frontend for a responsive, user-friendly experience.
- **Modular Design**: Organized into reusable modules (text processing, embeddings, retrieval, generation) for maintainability.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/tirth8205/RAG_using_NLP.git
   cd rag-using-nlp
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. **Authenticate with Hugging Face** (for gated models like `google/gemma-2b-it`):
   - Generate a Hugging Face token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Run:
     ```bash
     huggingface-cli login
     ```
     Paste your token when prompted.
4. **Run the Pipeline**:
   - Command-line:
     ```bash
     python rag_pipeline.py --pdf /path/to/your/file.pdf --query "Your question here"
     ```
   - Web UI:
     ```bash
     python app.py
     ```
     Open the provided URL (e.g., `http://127.0.0.1:8000`) in your browser, specify a Hugging Face model ID, upload a PDF, and ask a question.

## Requirements
- Python 3.11+
- ~10GB disk space (for embeddings and models)
- Optional: NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended for faster processing)

## Project Structure
- `rag_pipeline.py`: Main script for command-line RAG pipeline.
- `pdf_processor.py`: Extracts and chunks PDF text.
- `embedder.py`: Creates and manages embeddings.
- `retriever.py`: Retrieves relevant chunks using Faiss.
- `generator.py`: Generates answers with a user-specified Hugging Face LLM.
- `app.py`: FastAPI backend for the web UI.
- `templates/index.html`: Custom HTML/CSS frontend for the web UI.

## Example Output
```
Query: What is User Awareness?
Answer: User Awareness refers to understanding the needs, behaviors, and interactions of users with smart devices and environments in ubiquitous computing.
```

## Skills Demonstrated
- **NLP**: Text extraction, chunking, embeddings.
- **RAG**: Retrieval, augmentation, generation.
- **Machine Learning**: Sentence Transformers, Faiss, LLMs.
- **Software Engineering**: Modular design, error handling.
- **Web Development**: FastAPI backend, custom HTML/CSS/JavaScript frontend, API design.
- **GPU Optimization**: Local execution, batch processing.

## Future Improvements
- Add evaluation metrics for answer quality.
- Scale to multiple PDFs with a vector database.
- Support more model types (e.g., non-causal LLMs) in the generator.
- Add a loading indicator in the UI during processing.

## Author
- Tirth Kanani
- [LinkedIn](https://www.linkedin.com/in/tirthkanani/)
