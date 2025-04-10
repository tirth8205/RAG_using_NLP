# rag_pipeline.py
import argparse
from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings
from retriever import load_embeddings, retrieve_relevant_resources
from generator import load_llm, generate_answer
from typing import List, Dict

# Default constants
DEFAULT_PDF_PATH = 'Ubiquitous-Computing.pdf'
EMBEDDINGS_PATH = 'embeddings.csv'

def main(pdf_path: str, query: str):
    """Run the RAG pipeline to answer a query based on a user-specified PDF."""
    # Step 1: Extract and chunk PDF
    pages_and_chunks = extract_and_chunk_pdf(pdf_path)
    if not pages_and_chunks:
        print(f'Failed to process PDF at {pdf_path}.')
        return

    # Step 2: Create or load embeddings
    if not create_and_save_embeddings(pages_and_chunks, EMBEDDINGS_PATH):
        print('Failed to create embeddings.')
        return

    embeddings, chunk_list = load_embeddings(EMBEDDINGS_PATH)
    if embeddings is None:
        print('Failed to load embeddings.')
        return

    # Step 3: Retrieve relevant context
    scores, indices = retrieve_relevant_resources(query, embeddings)
    context_items = [chunk_list[i] for i in indices]

    # Step 4: Generate answer
    answer = generate_answer(query, context_items)
    print(f'Query: {query}')
    print(f'Answer: {answer}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a local RAG pipeline on a PDF.')
    parser.add_argument('--pdf', type=str, default=DEFAULT_PDF_PATH,
                        help='Path to the PDF file to process.')
    parser.add_argument('--query', type=str, default='What is User Awareness?',
                        help='The query to answer using the RAG pipeline.')
    args = parser.parse_args()
    main(args.pdf, args.query)
