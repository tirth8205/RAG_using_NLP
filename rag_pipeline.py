# rag_pipeline.py
import argparse
import os
import hashlib # For a more robust ID, though example uses filename
from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings, load_embeddings # Corrected: load_embeddings is used
from retriever import retrieve_relevant_resources
from generator import generate_answer
# from typing import List, Dict # No longer explicitly needed here, but good practice if types were complex

# Default constants
DEFAULT_PDF_PATH = 'Ubiquitous-Computing.pdf' # A default PDF if none provided
EMBEDDINGS_DIR_CLI = 'pdf_embeddings_cli'  # Directory for CLI embeddings
os.makedirs(EMBEDDINGS_DIR_CLI, exist_ok=True) # Ensure the directory exists

def get_pdf_id_from_path(pdf_path: str) -> str:
    """
    Generate an ID for the PDF. 
    Using filename for simplicity in CLI. For more robustness, use content hash.
    """
    # Simple approach: use filename (could be problematic if path changes but filename is same)
    # return os.path.basename(pdf_path) + ".csv" 
    
    # More robust: hash of the filename or content. Let's use filename hash for CLI.
    # Or, even better, hash of content if performance allows reading file early.
    # For this example, let's stick to a simpler filename-derived ID for CLI,
    # acknowledging its limitations.
    # A common pattern is to sanitize the filename to make it a valid part of a path.
    base_name = os.path.basename(pdf_path)
    # Replace problematic characters if any, though for CSV extension this is usually fine.
    return f"{base_name}.csv"


def main(pdf_path: str, query: str, model_id: str): # Added model_id to main function signature
    """Run the RAG pipeline to answer a query based on a user-specified PDF."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    pdf_filename_id = get_pdf_id_from_path(pdf_path) # e.g., "my_doc.pdf.csv"
    pdf_specific_embeddings_path = os.path.join(EMBEDDINGS_DIR_CLI, pdf_filename_id)
    print(f"Using PDF: {pdf_path}")
    print(f"Embeddings will be checked/stored at: {pdf_specific_embeddings_path}")

    # Step 1: Try to load existing embeddings
    embeddings, chunk_list = load_embeddings(pdf_specific_embeddings_path)

    if embeddings is None or not chunk_list:
        print(f"Embeddings not found at {pdf_specific_embeddings_path}. Processing PDF from scratch.")
        # Step 1.1: Extract and chunk PDF
        pages_and_chunks = extract_and_chunk_pdf(pdf_path)
        if not pages_and_chunks:
            print(f'Failed to process PDF at {pdf_path}.')
            return

        # Step 1.2: Create or load embeddings
        if not create_and_save_embeddings(pages_and_chunks, pdf_specific_embeddings_path):
            print('Failed to create and save embeddings.')
            return
        
        # Step 1.3: Load the newly created embeddings
        embeddings, chunk_list = load_embeddings(pdf_specific_embeddings_path)
        if embeddings is None or not chunk_list:
            print('Failed to load newly created embeddings.')
            return
    else:
        print("Successfully loaded existing embeddings.")

    # Step 2: Retrieve relevant context
    if embeddings.nelement() == 0:
        print("Error: Embeddings tensor is empty. Cannot retrieve context.")
        return
    scores, indices = retrieve_relevant_resources(query, embeddings)
    
    if indices.nelement() == 0:
        print("Could not find relevant information in the PDF for your query based on embeddings.")
        return

    context_items = [chunk_list[i] for i in indices.tolist()]

    # Step 3: Generate answer
    print(f"\nUsing LLM: {model_id} for generation.")
    answer = generate_answer(query, context_items, model_id) # Pass model_id
    print(f'\nQuery: {query}')
    print(f'Answer: {answer}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a local RAG pipeline on a PDF.')
    parser.add_argument('--pdf', type=str, default=DEFAULT_PDF_PATH,
                        help='Path to the PDF file to process.')
    parser.add_argument('--query', type=str, default='What is User Awareness?',
                        help='The query to answer using the RAG pipeline.')
    parser.add_argument('--model_id', type=str, default='google/gemma-2b-it', # Added model_id argument
                        help='Hugging Face Model ID for the generator (e.g., google/gemma-2b-it).')
    args = parser.parse_args()
    
    # Check if default PDF exists if no specific PDF is provided by user
    pdf_to_process = args.pdf
    if args.pdf == DEFAULT_PDF_PATH and not os.path.exists(DEFAULT_PDF_PATH):
        print(f"Warning: Default PDF '{DEFAULT_PDF_PATH}' not found. Please specify a PDF using --pdf.")
        # You might want to exit here if the default is critical and not found
        # For now, we'll let it proceed, and main() will catch the file not found.
        
    main(pdf_to_process, args.query, args.model_id)