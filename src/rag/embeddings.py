"""Reusable BGE-M3 inference; importing this module does not load a model."""

from functools import cached_property

import numpy as np


MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024


class BGEEmbedder:
    def __init__(self, *, device=None, max_length=512, revision=MODEL_REVISION):
        if type(max_length) is not int or not 0 < max_length <= 8192:
            raise ValueError("max_length must be between 1 and 8192")
        self.model_name = MODEL_NAME
        self.revision = revision
        self.dimension = EMBEDDING_DIMENSION
        self.max_length = max_length
        self.device = device

    @cached_property
    def model(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model = SentenceTransformer(self.model_name, revision=self.revision, device=device)
        model.max_seq_length = self.max_length
        if model.get_sentence_embedding_dimension() != self.dimension:
            raise ValueError("The embedding model does not produce 1024-dimensional vectors")
        return model

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def encode_passages(self, passages, *, batch_size=4):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        texts = list(passages)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Passages must be nonempty strings")
        lengths = self.tokenizer(
            texts, truncation=False, add_special_tokens=True, return_length=True
        )["length"]
        if any(length > self.max_length for length in lengths):
            raise ValueError("A passage exceeds max_length; chunk it before embedding")
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def encode_query(self, query):
        return self.encode_passages([query])[0]
