# Local RAG Pipeline using NLP for PDF Q&A

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **Natural Language Processing (NLP)** techniques to answer questions based on a user-specified PDF document. It runs locally on a GPU, processing text, generating embeddings, retrieving relevant chunks, and producing answers with a Large Language Model (LLM).

## Features
- **Custom PDF Support**: Users can upload any PDF via a command-line argument.
- **Local Execution**: Runs on your GPU (e.g., NVIDIA RTX 4090) for privacy and speed.
- **NLP-Driven**: Leverages text extraction, chunking, and embeddings for robust RAG.
- **Modular Design**: Organized into reusable modules for maintainability.

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/rag-using-nlp.git
   cd rag-using-nlp
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. **Run the Pipeline**:
   - Use the default PDF:
     ```bash
     python rag_pipeline.py --query "What is User Awareness?"
     ```
   - Or specify your own PDF:
     ```bash
     python rag_pipeline.py --pdf /path/to/your/file.pdf --query "Your question here"
     ```

## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
- ~10GB disk space (for embeddings and models)

## Project Structure
- `rag_pipeline.py`: Main script to run the pipeline.
- `pdf_processor.py`: Extracts and chunks PDF text.
- `embedder.py`: Creates and manages embeddings.
- `retriever.py`: Retrieves relevant chunks using Faiss.
- `generator.py`: Generates answers with an LLM.

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
- **GPU Optimization**: Local execution, batch processing.

## Future Improvements
- Add a Gradio UI for interactive PDF uploads and Q&A.
- Implement evaluation metrics for answer quality.
- Scale to multiple PDFs with a vector database.

## Author
- Tirth Kanani
- [LinkedIn](https://www.linkedin.com/in/tirthkanani/)

