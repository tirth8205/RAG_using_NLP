# embedder.py
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'

def create_and_save_embeddings(pages_and_chunks: List[Dict], save_path: str) -> bool:
    """Create embeddings for text chunks and save to CSV."""
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        text_chunks = [item['sentence_chunk'] for item in pages_and_chunks]
        embeddings = embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)
        
        for i, item in enumerate(pages_and_chunks):
            item['embedding'] = embeddings[i].cpu().numpy()

        df = pd.DataFrame(pages_and_chunks)
        df.to_csv(save_path, index=False)
        return True
    except Exception as e:
        print(f'Error creating embeddings: {e}')
        return False

def load_embeddings(embeddings_path: str) -> Tuple[torch.Tensor, List[Dict]]:
    """Load embeddings and chunk data from CSV."""
    try:
        df = pd.read_csv(embeddings_path)
        df['embedding'] = df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        embeddings = torch.tensor(np.stack(df['embedding'].values), dtype=torch.float32).to(DEVICE)
        return embeddings, df.to_dict(orient='records')
    except Exception as e:
        print(f'Error loading embeddings: {e}')
        return None, []

