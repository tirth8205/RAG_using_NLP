# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings, load_embeddings
from retriever import retrieve_relevant_resources
from generator import generate_answer
import os
import shutil

# Constants
EMBEDDINGS_PATH = 'embeddings.csv'

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_pdf(pdf_file: UploadFile = File(...), query: str = Form(...), model_id: str = Form(...)):
    """Process the uploaded PDF and generate an answer using the specified model."""
    print('Debug: Starting process_pdf_and_answer')
    print(f'Debug: pdf_file={pdf_file.filename}, query={query}, model_id={model_id}')
    
    if not pdf_file or not query or not model_id:
        print('Debug: Missing PDF, query, or model ID')
        return {"answer": "Please upload a PDF, enter a query, and specify a model ID."}
    
    # Save the uploaded PDF temporarily
    pdf_path = f"temp_{pdf_file.filename}"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)
    
    print(f'Debug: pdf_path={pdf_path}')
    if not pdf_path.lower().endswith('.pdf'):
        print('Debug: Invalid file type')
        os.remove(pdf_path)
        return {"answer": "Please upload a valid PDF file."}

    try:
        # Step 1: Extract and chunk PDF
        print('Debug: Extracting and chunking PDF')
        pages_and_chunks = extract_and_chunk_pdf(pdf_path)
        if not pages_and_chunks:
            print('Debug: Failed to process PDF')
            return {"answer": "Failed to process PDF."}

        # Step 2: Create embeddings
        print('Debug: Creating embeddings')
        if not create_and_save_embeddings(pages_and_chunks, EMBEDDINGS_PATH):
            print('Debug: Failed to create embeddings')
            return {"answer": "Failed to create embeddings."}

        # Step 3: Load embeddings
        print('Debug: Loading embeddings')
        embeddings, chunk_list = load_embeddings(EMBEDDINGS_PATH)
        if embeddings is None:
            print('Debug: Failed to load embeddings')
            return {"answer": "Failed to load embeddings."}

        # Step 4: Retrieve relevant context
        print('Debug: Retrieving relevant context')
        scores, indices = retrieve_relevant_resources(query, embeddings)
        context_items = [chunk_list[i] for i in indices]
        print(f'Debug: Retrieved {len(context_items)} context items')

        # Step 5: Generate answer using the specified model
        print('Debug: Generating answer')
        answer = generate_answer(query, context_items, model_id)
        print('Debug: Answer generated')
        return {"answer": answer}
    except Exception as e:
        print(f'Debug: Error occurred: {str(e)}')
        return {"answer": f"Error: {str(e)}"}
    finally:
        # Clean up the temporary PDF file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
