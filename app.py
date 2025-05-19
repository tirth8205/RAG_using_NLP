# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sse_starlette.sse import EventSourceResponse # Import this
import asyncio # For yielding control and async operations
import os
import shutil
import hashlib
import json # For sending structured data if needed

from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings, load_embeddings
from retriever import retrieve_relevant_resources
from generator import generate_answer

# Constants
EMBEDDINGS_DIR = 'pdf_embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_pdf_id(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def rag_pipeline_streamer(pdf_file_obj: UploadFile, query: str, model_id: str):
    """
    Generator function that performs the RAG pipeline steps and yields status updates.
    Each yielded item should be a dictionary like {"event": "event_name", "data": "message_string_or_json"}
    """
    temp_pdf_path = None
    try:
        yield {"event": "status", "data": "Processing started..."}
        await asyncio.sleep(0.01) # Allow message to be sent

        # --- File Handling and Validation ---
        # Use a safe filename and ensure it's a PDF
        original_filename = pdf_file_obj.filename if pdf_file_obj.filename else "uploaded_file"
        safe_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in original_filename)
        temp_pdf_path = f"temp_{safe_filename}"

        file_content = await pdf_file_obj.read()
        await pdf_file_obj.seek(0) # Reset pointer for saving

        with open(temp_pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file_obj.file, f)

        if not temp_pdf_path.lower().endswith('.pdf'):
            yield {"event": "error", "data": "Invalid file type. Please upload a PDF."}
            return

        # --- PDF ID and Embedding Check ---
        yield {"event": "status", "data": "Identifying PDF and checking existing embeddings..."}
        await asyncio.sleep(0.01)
        pdf_id = get_pdf_id(file_content)
        pdf_specific_embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{pdf_id}.csv")
        yield {"event": "status", "data": f"PDF ID: {pdf_id}"}
        await asyncio.sleep(0.01)

        embeddings, chunk_list = load_embeddings(pdf_specific_embeddings_path)

        if embeddings is None or not chunk_list:
            yield {"event": "status", "data": "Embeddings not found. Starting fresh processing..."}
            await asyncio.sleep(0.01)

            yield {"event": "status", "data": "Step 1/5: Extracting and chunking PDF content..."}
            await asyncio.sleep(0.01)
            # Run synchronous functions in a thread to avoid blocking the event loop
            pages_and_chunks = await asyncio.to_thread(extract_and_chunk_pdf, temp_pdf_path)
            if not pages_and_chunks:
                yield {"event": "error", "data": "Failed to extract text from PDF."}
                return
            yield {"event": "status", "data": "PDF content processed."}
            await asyncio.sleep(0.01)

            yield {"event": "status", "data": "Step 2/5: Creating and saving embeddings (this may take time for new models)..."}
            await asyncio.sleep(0.01)
            # This is where 'all-mpnet-base-v2' might download
            success_embedding = await asyncio.to_thread(create_and_save_embeddings, pages_and_chunks, pdf_specific_embeddings_path)
            if not success_embedding:
                yield {"event": "error", "data": "Failed to create and save embeddings."}
                return
            yield {"event": "status", "data": "Embeddings created successfully."}
            await asyncio.sleep(0.01)

            embeddings, chunk_list = load_embeddings(pdf_specific_embeddings_path) # Load them again
            if embeddings is None or not chunk_list:
                yield {"event": "error", "data": "Critical error: Failed to load newly created embeddings."}
                return
        else:
            yield {"event": "status", "data": "Existing embeddings found and loaded."}
            await asyncio.sleep(0.01)

        if embeddings.nelement() == 0: # Check if tensor is empty
            yield {"event": "error", "data": "Embeddings data is empty. Cannot proceed."}
            return

        # --- Retrieval ---
        yield {"event": "status", "data": "Step 3/5: Retrieving relevant context..."}
        await asyncio.sleep(0.01)
        scores, indices = await asyncio.to_thread(retrieve_relevant_resources, query, embeddings)
        if indices.nelement() == 0:
            yield {"event": "final_answer", "data": "Could not find relevant information in the PDF for your query."}
            yield {"event": "status", "data": "No relevant context found."}
            return
        context_items = [chunk_list[i] for i in indices.tolist()]
        yield {"event": "status", "data": f"Retrieved {len(context_items)} relevant snippets."}
        await asyncio.sleep(0.01)

        # --- Generation ---
        yield {"event": "status", "data": "Step 4/5: Loading Language Model (this may take time for new models)..."}
        await asyncio.sleep(0.01)
        # The generate_answer function internally loads the LLM like 'google/gemma-2b-it'
        yield {"event": "status", "data": "Step 5/5: Generating answer..."}
        await asyncio.sleep(0.01)
        answer = await asyncio.to_thread(generate_answer, query, context_items, model_id)

        yield {"event": "status", "data": "Answer generated!"}
        await asyncio.sleep(0.01)
        yield {"event": "final_answer", "data": answer}

    except HTTPException as http_exc: # To catch deliberate HTTP errors if any
        yield {"event": "error", "data": f"Error: {http_exc.detail}"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Unexpected error in RAG pipeline streamer: {error_details}")
        yield {"event": "error", "data": f"An unexpected server error occurred: {str(e)}"}
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as e_rem:
                print(f"Error removing temp file {temp_pdf_path}: {e_rem}")
        yield {"event": "status", "data": "Processing complete."}
        # No more yields after this. Client should handle stream end.

@app.post("/stream-process") # New endpoint for SSE
async def stream_process_pdf_endpoint(
    pdf_file: UploadFile = File(...),
    query: str = Form(...),
    model_id: str = Form(...)
):
    # Directly pass the UploadFile object to the streamer
    return EventSourceResponse(rag_pipeline_streamer(pdf_file, query, model_id))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)