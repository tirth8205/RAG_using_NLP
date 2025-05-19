# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
import asyncio
import os
import shutil
import hashlib
import json
from pydantic import BaseModel # For request body of download endpoint

from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings, load_embeddings
from retriever import retrieve_relevant_resources
# We will modify generator.py, so ensure it's updated when you use this app.py
import generator # Changed to import the module

# Constants
EMBEDDINGS_DIR = 'pdf_embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_pdf_id(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: FastAPIRequest): # Renamed to avoid conflict with pydantic's Request
    return templates.TemplateResponse("index.html", {"request": request})


async def rag_pipeline_streamer(
    pdf_file_obj: UploadFile,
    query: str,
    llm_service: str,
    hf_model_id: str | None = None, # Can be None if API service is used
    api_key: str | None = None,     # Can be None if local HF model is used
    api_model_name: str | None = None # Can be None if local HF model is used
):
    temp_pdf_path = None
    try:
        yield {"event": "status", "data": "Processing started..."}
        await asyncio.sleep(0.01)

        original_filename = pdf_file_obj.filename if pdf_file_obj.filename else "uploaded_file"
        safe_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in original_filename)
        temp_pdf_path = f"temp_{safe_filename}"
        file_content = await pdf_file_obj.read()
        await pdf_file_obj.seek(0)
        with open(temp_pdf_path, "wb") as f: shutil.copyfileobj(pdf_file_obj.file, f)

        if not temp_pdf_path.lower().endswith('.pdf'):
            yield {"event": "error", "data": "Invalid file type. Please upload a PDF."}
            return

        yield {"event": "status", "data": "Identifying PDF and checking existing embeddings..."}
        pdf_id = get_pdf_id(file_content)
        pdf_specific_embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{pdf_id}.csv")
        yield {"event": "status", "data": f"PDF ID: {pdf_id}"}; await asyncio.sleep(0.01)

        embeddings, chunk_list = await asyncio.to_thread(load_embeddings, pdf_specific_embeddings_path)

        if embeddings is None or not chunk_list:
            yield {"event": "status", "data": "Embeddings not found. Starting fresh processing..."}; await asyncio.sleep(0.01)
            yield {"event": "status", "data": "Step 1/5: Extracting and chunking PDF content..."}; await asyncio.sleep(0.01)
            pages_and_chunks = await asyncio.to_thread(extract_and_chunk_pdf, temp_pdf_path)
            if not pages_and_chunks:
                yield {"event": "error", "data": "Failed to extract text from PDF."}; return
            yield {"event": "status", "data": "PDF content processed."}; await asyncio.sleep(0.01)

            yield {"event": "status", "data": "Step 2/5: Creating/saving embeddings (may download embedding model)..."}; await asyncio.sleep(0.01)
            success_embedding = await asyncio.to_thread(create_and_save_embeddings, pages_and_chunks, pdf_specific_embeddings_path)
            if not success_embedding:
                yield {"event": "error", "data": "Failed to create/save embeddings."}; return
            yield {"event": "status", "data": "Embeddings created."}; await asyncio.sleep(0.01)
            embeddings, chunk_list = await asyncio.to_thread(load_embeddings, pdf_specific_embeddings_path)
            if embeddings is None or not chunk_list:
                yield {"event": "error", "data": "Failed to load newly created embeddings."}; return
        else:
            yield {"event": "status", "data": "Existing embeddings loaded."}; await asyncio.sleep(0.01)

        if embeddings.nelement() == 0:
            yield {"event": "error", "data": "Embeddings data is empty."}; return

        yield {"event": "status", "data": "Step 3/5: Retrieving relevant context..."}; await asyncio.sleep(0.01)
        scores, indices = await asyncio.to_thread(retrieve_relevant_resources, query, embeddings)
        if indices.nelement() == 0:
            yield {"event": "final_answer", "data": "Could not find relevant information in the PDF for the query."}
            yield {"event": "status", "data": "No relevant context found."}; return
        context_items = [chunk_list[i] for i in indices.tolist()]
        yield {"event": "status", "data": f"Retrieved {len(context_items)} relevant snippets."}; await asyncio.sleep(0.01)

        yield {"event": "status", "data": f"Step 4/5: Initializing LLM service: {llm_service.upper()}..."}; await asyncio.sleep(0.01)
        if llm_service != "huggingface":
             yield {"event": "status", "data": f"Using model: {api_model_name}"}; await asyncio.sleep(0.01)

        yield {"event": "status", "data": "Step 5/5: Generating answer (may download LLM if local)..."}; await asyncio.sleep(0.01)
        answer = await asyncio.to_thread(
            generator.generate_answer, # Call the function from the imported module
            query,
            context_items,
            llm_service=llm_service,
            hf_model_id=hf_model_id,
            api_key=api_key,
            api_model_name=api_model_name
        )

        yield {"event": "status", "data": "Answer generated!"}; await asyncio.sleep(0.01)
        yield {"event": "final_answer", "data": answer}

    except HTTPException as http_exc:
        yield {"event": "error", "data": f"Error: {http_exc.detail}"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Unexpected error in RAG pipeline streamer: {error_details}")
        yield {"event": "error", "data": f"An unexpected server error occurred: {str(e)}"}
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try: os.remove(temp_pdf_path)
            except Exception as e_rem: print(f"Error removing temp file {temp_pdf_path}: {e_rem}")
        yield {"event": "status", "data": "Processing complete."}


@app.post("/stream-process")
async def stream_process_pdf_endpoint(
    pdf_file: UploadFile = File(...),
    query: str = Form(...),
    llm_service: str = Form(...),
    hf_model_id: str = Form(None), # Use None as default for optional fields
    api_key: str = Form(None),
    api_model_name: str = Form(None)
):
    return EventSourceResponse(rag_pipeline_streamer(
        pdf_file, query, llm_service, hf_model_id, api_key, api_model_name
    ))

class HFModelDownloadRequest(BaseModel):
    model_id: str

async def hf_model_downloader_streamer(model_id: str):
    """Streams messages about the Hugging Face model download/caching process."""
    yield {"event": "status", "data": f"Attempting to download/cache Hugging Face model: {model_id}..."}
    await asyncio.sleep(0.01)
    try:
        # These calls will trigger download if not cached and print to stdout/stderr.
        # Capturing that for fine-grained SSE is complex.
        # We send general status messages.
        from transformers import AutoTokenizer, AutoModelForCausalLM # Import here to keep other parts cleaner

        yield {"event": "status", "data": f"Downloading/caching tokenizer for {model_id}..."}
        await asyncio.to_thread(AutoTokenizer.from_pretrained, model_id)
        yield {"event": "status", "data": f"Tokenizer for {model_id} cached."}
        await asyncio.sleep(0.01)

        yield {"event": "status", "data": f"Downloading/caching model for {model_id} (this may take a while)..."}
        await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_id)
        yield {"event": "status", "data": f"Model {model_id} cached successfully."}

    except ImportError:
        yield {"event": "error", "data": "Transformers library not found. Please install it."}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error downloading/caching HF model {model_id}: {error_details}")
        yield {"event": "error", "data": f"Failed to download/cache model {model_id}: {str(e)}"}
    finally:
        yield {"event": "status", "data": f"Finished attempt to cache {model_id}."}

@app.post("/download-hf-model")
async def download_hf_model_endpoint(request: HFModelDownloadRequest):
    return EventSourceResponse(hf_model_downloader_streamer(request.model_id))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    