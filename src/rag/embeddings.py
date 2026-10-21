import torch
from sentence_transformers import SentenceTransformer

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = SentenceTransformer("BAAI/bge-m3", device=device)
model.max_seq_length = 512

passages = []

vectors = model.encode(
    passages,
    batch_size=4,
    normalize_embeddings=True,
)

query_vector = model.encode(
    [],
    normalize_embeddings=True,
)

print(vectors.shape)  # (2, 1024)