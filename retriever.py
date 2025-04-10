# retriever.py
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import Tuple

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
EMBEDDING_DIM = 768

def retrieve_relevant_resources(query: str, embeddings: torch.Tensor, n_resources: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retrieve top-k relevant chunks based on query."""
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        query_embedding = embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings.cpu().numpy())
        distances, indices = index.search(query_embedding, n_resources)

        scores = torch.tensor(distances[0], dtype=torch.float32)
        indices = torch.tensor(indices[0], dtype=torch.int64)
        return scores, indices
    except Exception as e:
        print(f'Error in retrieval: {e}')
        return torch.tensor([]), torch.tensor([])

