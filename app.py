# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request as FastAPIRequest
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
import asyncio
import os
import shutil
import hashlib
from pydantic import BaseModel # For request body of download endpoint
import tempfile # For creating a uniquely named temporary file more robustly

from pdf_processor import extract_and_chunk_pdf
from embedder import create_and_save_embeddings, load_embeddings
from retriever import retrieve_relevant_resources
import generator # Import the module

# Constants
EMBEDDINGS_DIR = 'pdf_embeddings'
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_pdf_id(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: FastAPIRequest):
    return templates.TemplateResponse("index.html", {"request": request})


async def rag_pipeline_streamer(
    # pdf_file_obj: UploadFile, <--- NO LONGER PASSING UploadFile OBJECT
    saved_pdf_path: str,         # <--- PASSING PATH TO OUR SAVED TEMP FILE
    pdf_content_for_hash: bytes, # <--- PASSING CONTENT FOR HASH
    query: str,
    llm_service: str,
    hf_model_id: str | None = None,
    api_key: str | None = None,
    api_model_name: str | None = None
):
    # temp_pdf_path is now saved_pdf_path, managed by the caller
    print(f"DEBUG: rag_pipeline_streamer CALLED with saved_pdf_path: {saved_pdf_path}")
    print(f"DEBUG: Query='{query}', LLM_Service='{llm_service}', HF_Model_ID='{hf_model_id}', API_Key_Present={'Yes' if api_key else 'No'}, API_Model_Name='{api_model_name}'")

    try:
        yield {"event": "status", "data": "Processing started (streamer)..."}
        print("DEBUG: Streamer - Yielded 'Processing started...'")
        await asyncio.sleep(0.01)

        # File content is already read and path is provided.
        # Hashing uses pdf_content_for_hash.
        # PDF processing uses saved_pdf_path.

        yield {"event": "status", "data": "Identifying PDF and checking existing embeddings..."}
        print("DEBUG: Streamer - Yielded 'Identifying PDF...'")
        pdf_id = get_pdf_id(pdf_content_for_hash) # Hash is from the content we reliably read
        pdf_specific_embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{pdf_id}.csv")
        yield {"event": "status", "data": f"PDF ID: {pdf_id}"}; print(f"DEBUG: Streamer - PDF ID is {pdf_id}"); await asyncio.sleep(0.01)

        embeddings, chunk_list = await asyncio.to_thread(load_embeddings, pdf_specific_embeddings_path)
        print(f"DEBUG: Streamer - load_embeddings result - Embeddings type: {type(embeddings)}, Chunk list length: {len(chunk_list) if chunk_list else 'N/A'}")

        if embeddings is None or not chunk_list:
            yield {"event": "status", "data": "Embeddings not found. Starting fresh processing..."}; print("DEBUG: Streamer - Embeddings not found...") ;await asyncio.sleep(0.01)
            yield {"event": "status", "data": "Step 1/5: Extracting and chunking PDF content..."}; print("DEBUG: Streamer - Starting PDF chunking..."); await asyncio.sleep(0.01)
            # Pass the path of YOUR temporary file to pdf_processor
            pages_and_chunks = await asyncio.to_thread(extract_and_chunk_pdf, saved_pdf_path) # Use saved_pdf_path
            if not pages_and_chunks:
                print("ERROR: Streamer - Failed to extract text from PDF (pages_and_chunks is empty/None)")
                yield {"event": "error", "data": "Failed to extract text from PDF."}; return
            yield {"event": "status", "data": "PDF content processed."}; print("DEBUG: Streamer - PDF chunking done."); await asyncio.sleep(0.01)

            yield {"event": "status", "data": "Step 2/5: Creating/saving embeddings (may download embedding model)..."}; print("DEBUG: Streamer - Creating embeddings..."); await asyncio.sleep(0.01)
            success_embedding = await asyncio.to_thread(create_and_save_embeddings, pages_and_chunks, pdf_specific_embeddings_path)
            if not success_embedding:
                print("ERROR: Streamer - Failed to create/save embeddings")
                yield {"event": "error", "data": "Failed to create/save embeddings."}; return
            yield {"event": "status", "data": "Embeddings created."}; print("DEBUG: Streamer - Embeddings created."); await asyncio.sleep(0.01)

            embeddings, chunk_list = await asyncio.to_thread(load_embeddings, pdf_specific_embeddings_path)
            print(f"DEBUG: Streamer - Reloaded embeddings - Embeddings type: {type(embeddings)}, Chunk list length: {len(chunk_list) if chunk_list else 'N/A'}")
            if embeddings is None or not chunk_list:
                print("ERROR: Streamer - Failed to load newly created embeddings")
                yield {"event": "error", "data": "Critical error: Failed to load newly created embeddings."}; return
        else:
            yield {"event": "status", "data": "Existing embeddings found and loaded."}; print("DEBUG: Streamer - Existing embeddings loaded."); await asyncio.sleep(0.01)

        if not hasattr(embeddings, 'nelement') or embeddings.nelement() == 0:
            print("ERROR: Streamer - Embeddings tensor is invalid or empty")
            yield {"event": "error", "data": "Embeddings data is invalid or empty. Cannot proceed."}; return

        yield {"event": "status", "data": "Step 3/5: Retrieving relevant context..."}; print("DEBUG: Streamer - Retrieving context..."); await asyncio.sleep(0.01)
        scores, indices = await asyncio.to_thread(retrieve_relevant_resources, query, embeddings)
        if not hasattr(indices, 'nelement') or indices.nelement() == 0:
            print("INFO: Streamer - No relevant context found for query.")
            yield {"event": "final_answer", "data": "Could not find relevant information in the PDF for your query."}
            yield {"event": "status", "data": "No relevant context found."}; return

        context_items = [chunk_list[i] for i in indices.tolist()]
        yield {"event": "status", "data": f"Retrieved {len(context_items)} relevant snippets."}; print(f"DEBUG: Streamer - Retrieved {len(context_items)} snippets."); await asyncio.sleep(0.01)

        yield {"event": "status", "data": f"Step 4/5: Initializing LLM service: {llm_service.upper()}..."}; print(f"DEBUG: Streamer - Initializing LLM: {llm_service}"); await asyncio.sleep(0.01)
        if llm_service != "huggingface" and api_model_name:
             yield {"event": "status", "data": f"Using model: {api_model_name}"}; print(f"DEBUG: Streamer - API Model: {api_model_name}"); await asyncio.sleep(0.01)

        yield {"event": "status", "data": "Step 5/5: Generating answer (this may take time if models need to download)..."}; print("DEBUG: Streamer - Calling generator.generate_answer"); await asyncio.sleep(0.01)
        answer = await asyncio.to_thread(
            generator.generate_answer,
            query,
            context_items,
            llm_service=llm_service,
            hf_model_id=hf_model_id,
            api_key=api_key,
            api_model_name=api_model_name
        )
        print(f"DEBUG: Streamer - Answer from generator: '{str(answer)[:200]}...'")

        if isinstance(answer, str) and answer.startswith("Error:"):
            yield {"event": "error", "data": answer}
        else:
            yield {"event": "final_answer", "data": answer}
        yield {"event": "status", "data": "Answer generation attempt complete."} ; print("DEBUG: Streamer - Yielded 'Answer generation attempt complete.'"); await asyncio.sleep(0.01)

    except HTTPException as http_exc: # Should not happen here if all inputs are validated by caller
        print(f"ERROR: Streamer - HTTPException in streamer: {http_exc.detail}")
        yield {"event": "error", "data": f"Error: {http_exc.detail}"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"CRITICAL UNEXPECTED ERROR in RAG pipeline streamer: {error_details}")
        yield {"event": "error", "data": f"An unexpected server error occurred in streamer: {str(e)}"}
    finally:
        print("DEBUG: Streamer - rag_pipeline_streamer finally block executing.")
        # The temporary file (saved_pdf_path) is now managed and deleted by the caller endpoint
        yield {"event": "status", "data": "Processing complete (streamer)."}
        print("DEBUG: Streamer - Yielded 'Processing complete.' Stream should now close.")


@app.post("/stream-process")
async def stream_process_pdf_endpoint(
    pdf_file: UploadFile = File(...),
    query: str = Form(...),
    llm_service: str = Form(...),
    hf_model_id: str = Form(None),
    api_key: str = Form(None),
    api_model_name: str = Form(None)
):
    """Endpoint to process PDF and query, streaming results back."""
    print(f"ENDPOINT: /stream-process CALLED")
    print(f"ENDPOINT: Query='{query}', LLM_Service='{llm_service}', HF_Model_ID='{hf_model_id}', API_Key_Present={'Yes' if api_key else 'No'}, API_Model_Name='{api_model_name}'")

    # Create a secure temporary file path
    # Use a more robust way to create a temporary file name and path
    original_filename = pdf_file.filename if pdf_file.filename else "uploaded_file.pdf"
    safe_suffix = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in original_filename)
    if not safe_suffix.lower().endswith('.pdf'):
        safe_suffix += ".pdf"

    # Using tempfile module for safer temporary file creation
    temp_file_descriptor, temp_pdf_path = tempfile.mkstemp(suffix=f"_{safe_suffix}", prefix="rag_app_")
    os.close(temp_file_descriptor) # We'll open it in 'wb' mode ourselves

    print(f"ENDPOINT: Temp PDF Path created: {temp_pdf_path}")
    file_content_for_hash = b''

    try:
        # --- Save UploadFile immediately to our own temp file ---
        print(f"ENDPOINT: Attempting to save uploaded file to {temp_pdf_path}")
        try:
            # Read the content from UploadFile. This is the critical part.
            # We do this only once.
            contents = await pdf_file.read()
            if not contents:
                print("ERROR ENDPOINT: UploadFile.read() returned empty content.")
                # This case should ideally raise an HTTPException or return an error response
                # For SSE, it's tricky. Let's let the streamer handle yielding an error.
                # However, to prevent further processing with empty content:
                raise HTTPException(status_code=400, detail="Uploaded PDF file appears to be empty.")

            with open(temp_pdf_path, "wb") as f_out:
                f_out.write(contents)
            file_content_for_hash = contents # Use the read contents for hashing
            print(f"ENDPOINT: UploadFile successfully saved to {temp_pdf_path}. Size: {len(file_content_for_hash)} bytes.")

        except ValueError as ve: # Catch "read of closed file" or "seek of closed file" specifically
            print(f"ERROR ENDPOINT: ValueError during UploadFile read/seek: {ve}")
            import traceback
            traceback.print_exc()
            # Propagate this as an HTTP error, SSE won't even start properly.
            raise HTTPException(status_code=500, detail=f"Error processing uploaded file: {str(ve)}")
        except Exception as e_file_save:
            print(f"ERROR ENDPOINT: During initial saving of uploaded PDF to {temp_pdf_path}: {e_file_save}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded PDF file: {str(e_file_save)}")
        finally:
            # Ensure UploadFile is closed by Starlette, or try to close it.
            if hasattr(pdf_file, 'close') and callable(pdf_file.close):
                try:
                    if asyncio.iscoroutinefunction(pdf_file.close): await pdf_file.close()
                    else: await asyncio.to_thread(pdf_file.close)
                    print("DEBUG ENDPOINT: Called pdf_file.close()")
                except Exception as e_close:
                    print(f"DEBUG ENDPOINT: Error trying to close pdf_file: {e_close}")
        
        # Now, call the streamer with the path to the saved file and its content
        return EventSourceResponse(rag_pipeline_streamer(
            saved_pdf_path=temp_pdf_path,
            pdf_content_for_hash=file_content_for_hash,
            query=query,
            llm_service=llm_service,
            hf_model_id=hf_model_id,
            api_key=api_key,
            api_model_name=api_model_name
            # The temp_pdf_path needs to be cleaned up after the streamer is done.
            # This is tricky because EventSourceResponse is fire-and-forget.
            # A background task or a more complex setup is needed for robust cleanup
            # if the streamer itself doesn't delete it.
            # For now, the streamer's finally block will try, but it won't be called
            # if this endpoint function errors out before returning the EventSourceResponse.
            # The current streamer deletes 'saved_pdf_path' in its finally block.
        ))
    except HTTPException as http_e: # Re-raise HTTPExceptions
        if os.path.exists(temp_pdf_path): # Clean up if error happens before streamer starts
            try: os.remove(temp_pdf_path)
            except: pass
        raise http_e
    except Exception as e: # Catch any other unexpected errors in the endpoint
        print(f"CRITICAL UNEXPECTED ERROR in /stream-process endpoint: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_pdf_path): # Clean up
            try: os.remove(temp_pdf_path)
            except: pass
        # Return a generic error as SSE might not even start
        # This ideally should be an HTTPException too, but the return type is EventSourceResponse
        # For now, log it. The client will likely see a broken stream.
        # A better approach would be to return a simple JSON error response if we can't even start SSE.
        # This part needs more robust error handling if SSE setup itself fails.
        # For now, if we reach here, the client might just see a broken connection.
        raise HTTPException(status_code=500, detail="Server error before streaming could start.")


class HFModelDownloadRequest(BaseModel):
    model_id: str

async def hf_model_downloader_streamer(model_id: str):
    """Streams status of Hugging Face model download/caching process."""
    yield {"event": "status", "data": f"Attempting to download/cache Hugging Face model: {model_id}..."}
    await asyncio.sleep(0.01)
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

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
    """Endpoint to trigger download/caching of a Hugging Face model."""
    return EventSourceResponse(hf_model_downloader_streamer(request.model_id))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)