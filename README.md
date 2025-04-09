# RAG using NLP

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Natural Language Processing (NLP) techniques to enable question-answering on PDF documents. It runs locally on a GPU and uses an open-source textbook on ubiquitous computing as a sample dataset.

## Features
- Extracts and preprocesses text from PDFs using NLP tools like spaCy.
- Embeds text chunks using a pre-trained model (`all-mpnet-base-v2`).
- Implements a retrieval system for relevant content.
- Generates answers with a Large Language Model (LLM).

## Requirements
- Python 3.8+
- NVIDIA GPU (tested on RTX 4090) or equivalent
- Libraries: PyMuPDF, spaCy, sentence-transformers, torch, tqdm, accelerate, bitsandbytes, flash-attn
- See `RAG_using_NLP.ipynb` for detailed setup instructions.

## Usage
1. Clone this repository: `git clone https://github.com/your-username/rag-using-nlp.git`
2. Install dependencies (refer to the notebook).
3. Run `RAG_using_NLP.ipynb` to process a PDF and start querying.

## Project Structure
- `RAG_using_NLP.ipynb`: Main notebook with the RAG pipeline implementation.
- `README.md`: Project overview and instructions.

## License
This project is licensed under the MIT License.