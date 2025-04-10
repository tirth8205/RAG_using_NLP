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
    print('Debug: Starting process_pdf_and_answer')
    print(f'Debug: pdf_file={pdf_file}, query={query}')
    
    if not pdf_file or not query:
        print('Debug: Missing PDF or query')
        return 'Please upload a PDF and enter a query.'
    
    # Gradio 3.50.2 passes pdf_file as a file path string or file object
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    print(f'Debug: pdf_path={pdf_path}')
    if not pdf_path.lower().endswith('.pdf'):
        print('Debug: Invalid file type')
        return 'Please upload a valid PDF file.'

    try:
        # Step 1: Extract and chunk PDF
        print('Debug: Extracting and chunking PDF')
        pages_and_chunks = extract_and_chunk_pdf(pdf_path)
        if not pages_and_chunks:
            print('Debug: Failed to process PDF')
            return 'Failed to process PDF.'

        # Step 2: Create embeddings
        print('Debug: Creating embeddings')
        if not create_and_save_embeddings(pages_and_chunks, EMBEDDINGS_PATH):
            print('Debug: Failed to create embeddings')
            return 'Failed to create embeddings.'

        # Step 3: Load embeddings
        print('Debug: Loading embeddings')
        embeddings, chunk_list = load_embeddings(EMBEDDINGS_PATH)
        if embeddings is None:
            print('Debug: Failed to load embeddings')
            return 'Failed to load embeddings.'

        # Step 4: Retrieve relevant context
        print('Debug: Retrieving relevant context')
        scores, indices = retrieve_relevant_resources(query, embeddings)
        context_items = [chunk_list[i] for i in indices]
        print(f'Debug: Retrieved {len(context_items)} context items')

        # Step 5: Generate answer
        print('Debug: Generating answer')
        answer = generate_answer(query, context_items)
        print('Debug: Answer generated')
        return answer
    except Exception as e:
        print(f'Debug: Error occurred: {str(e)}')
        return f'Error: {str(e)}'

# Custom CSS
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #6b48ff, #00ddeb) !important;
    font-family: 'Poppins', sans-serif !important;
    color: #333 !important;
    margin: 0 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
}
h1 {
    color: #ffffff !important;
    text-align: center !important;
    font-size: 2.8em !important;
    margin-bottom: 10px !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2) !important;
}
.description {
    text-align: center !important;
    font-size: 1.3em !important;
    color: #e0e0e0 !important;
    margin-bottom: 30px !important;
}
.input-container, .output-container {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    margin: 15px auto !important;
    max-width: 700px !important;
    width: 90% !important;
}
.input-container label, .output-container label {
    font-weight: 600 !important;
    color: #2c3e50 !important;
    font-size: 1.2em !important;
    margin-bottom: 10px !important;
    display: block !important;
}
.input-container input[type='file'], .input-container textarea {
    border: 2px solid #ddd !important;
    border-radius: 10px !important;
    padding: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: #f9f9f9 !important;
    transition: border-color 0.3s ease !important;
}
.input-container input[type='file']:hover, .input-container textarea:hover {
    border-color: #6b48ff !important;
}
.input-container textarea {
    height: 120px !important;
    resize: none !important;
}
.output-container textarea {
    border: 2px solid #ddd !important;
    border-radius: 10px !important;
    padding: 12px !important;
    width: 100% !important;
    height: 180px !important;
    box-sizing: border-box !important;
    background: #f9f9f9 !important;
}
.submit-btn {
    background: linear-gradient(45deg, #ff6b6b, #ff8e53) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 40px !important;
    font-size: 1.2em !important;
    cursor: pointer !important;
    transition: transform 0.2s ease, background 0.3s ease !important;
    display: block !important;
    margin: 20px auto !important;
}
.submit-btn:hover {
    background: linear-gradient(45deg, #ff8e53, #ff6b6b) !important;
    transform: scale(1.05) !important;
}
.clear-btn {
    background: #e0e0e0 !important;
    color: #333 !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 40px !important;
    font-size: 1.2em !important;
    cursor: pointer !important;
    transition: background 0.3s ease !important;
    margin-right: 10px !important;
}
.clear-btn:hover {
    background: #d0d0d0 !important;
}
.button-container {
    text-align: center !important;
    margin-top: 20px !important;
}
.footer {
    text-align: center !important;
    font-size: 1.1em !important;
    color: #ffffff !important;
    margin-top: 40px !important;
    padding: 15px !important;
    background: rgba(0, 0, 0, 0.1) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
}
.footer span {
    color: #ff6b6b !important;
    font-weight: bold !important;
}
</style>
"""

# Gradio Blocks layout for 3.50.2
with gr.Blocks() as interface:
    gr.HTML(custom_css)  # Inject CSS
    gr.Markdown("# RAG using NLP: PDF Q&A")
    gr.Markdown("Upload a PDF and ask a question to get answers based on its content.")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF")
            query_input = gr.Textbox(label="Your Question", placeholder="e.g., What is User Awareness?")
            with gr.Row():
                clear_btn = gr.Button("Clear", elem_classes="clear-btn")
                submit_btn = gr.Button("Submit", elem_classes="submit-btn")
    
    output = gr.Textbox(label="Answer")
    
    # Event handling
    submit_btn.click(
        fn=process_pdf_and_answer,
        inputs=[pdf_input, query_input],
        outputs=output
    )
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[pdf_input, query_input]
    )
    
    gr.Markdown('<div class="footer">Made with ❤️ by <span>Tirth</span></div>')

# Launch the app
interface.launch()