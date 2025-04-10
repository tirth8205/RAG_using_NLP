# app.py
import gradio as gr
from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings, load_embeddings
from retriever import retrieve_relevant_resources
from generator import generate_answer
import os

# Constants
EMBEDDINGS_PATH = 'embeddings.csv'

def process_pdf_and_answer(pdf_file, query):
    """Process the uploaded PDF and generate an answer to the query."""
    if not pdf_file or not query:
        return 'Please upload a PDF and enter a query.'

    pdf_path = pdf_file.name  # Gradio uploads file to temp location
    try:
        # Step 1: Extract and chunk PDF
        pages_and_chunks = extract_and_chunk_pdf(pdf_path)
        if not pages_and_chunks:
            return 'Failed to process PDF.'

        # Step 2: Create embeddings
        if not create_and_save_embeddings(pages_and_chunks, EMBEDDINGS_PATH):
            return 'Failed to create embeddings.'

        # Step 3: Load embeddings
        embeddings, chunk_list = load_embeddings(EMBEDDINGS_PATH)
        if embeddings is None:
            return 'Failed to load embeddings.'

        # Step 4: Retrieve relevant context
        scores, indices = retrieve_relevant_resources(query, embeddings)
        context_items = [chunk_list[i] for i in indices]

        # Step 5: Generate answer
        answer = generate_answer(query, context_items)
        return answer
    except Exception as e:
        return f'Error: {str(e)}'

# Gradio interface
with gr.Blocks(title='RAG using NLP: PDF Q&A') as demo:
    gr.Markdown('# RAG using NLP: PDF Q&A')
    gr.Markdown('Upload a PDF and ask a question to get answers based on its content.')
    with gr.Row():
        pdf_input = gr.File(label='Upload PDF', file_types=['.pdf'])
        query_input = gr.Textbox(label='Your Question', placeholder='e.g., What is User Awareness?')
    submit_btn = gr.Button('Submit')
    output = gr.Textbox(label='Answer', lines=5)
    
    submit_btn.click(
        fn=process_pdf_and_answer,
        inputs=[pdf_input, query_input],
        outputs=output
    )

# Launch the app
demo.launch()

