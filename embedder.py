# embedder.py
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' # Consider making this configurable if different models are needed per PDF

def create_and_save_embeddings(pages_and_chunks: List[Dict], file_path: str) -> bool:
    """Create embeddings for text chunks and save to a specific CSV file."""
    try:
        # Ensure the directory for the file_path exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        text_chunks = [item['sentence_chunk'] for item in pages_and_chunks]
        
        print(f"Generating embeddings for {len(text_chunks)} chunks using {EMBEDDING_MODEL_NAME} on {DEVICE}...")
        embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
        
        for i, item in enumerate(pages_and_chunks):
            item['embedding'] = embeddings[i].cpu().numpy()

        df = pd.DataFrame(pages_and_chunks)
        df.to_csv(file_path, index=False)
        print(f"Embeddings saved to {file_path}")
        return True
    except Exception as e:
        print(f'Error creating and saving embeddings at {file_path}: {e}')
        return False

def load_embeddings(file_path: str) -> Tuple[torch.Tensor, List[Dict]]:
    """Load embeddings and chunk data from a specific CSV file."""
    try:
        if not os.path.exists(file_path):
            print(f"Embeddings file not found: {file_path}")
            return None, []
            
        df = pd.read_csv(file_path)
        # Convert string representation of embedding back to numpy array
        df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        
        # Ensure all embeddings have a consistent length (handle potential partial reads/writes if any)
        # This is a basic check; more robust validation might be needed depending on data integrity concerns
        first_embedding_len = len(df['embedding'].iloc[0]) if not df.empty else 0
        if not all(len(emb) == first_embedding_len for emb in df['embedding']):
            print(f"Warning: Inconsistent embedding lengths found in {file_path}. This might cause issues.")
            # Optionally, decide how to handle this: error out, try to filter, etc.
            # For now, we'll proceed but this indicates a potential issue with the CSV.

        embeddings_array = np.stack(df['embedding'].values)
        embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32).to(DEVICE)
        
        # Remove the 'embedding' column before converting to dict to avoid large data in chunk_list
        chunk_data_df = df.drop(columns=['embedding'])
        chunk_list = chunk_data_df.to_dict(orient='records')
        
        print(f"Embeddings loaded from {file_path}")
        return embeddings_tensor, chunk_list
    except Exception as e:
        print(f'Error loading embeddings from {file_path}: {e}')
        return None, []