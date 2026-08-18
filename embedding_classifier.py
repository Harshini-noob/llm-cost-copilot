from sentence_transformers import SentenceTransformer
import numpy as np
from test_prompts import TEST_PROMPTS

# Load once - this is a real (small) pretrained model, downloaded on first run
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute embeddings for your labeled reference set, once at import time
_reference_prompts = [item["prompt"] for item in TEST_PROMPTS]
_reference_tiers = [item["expected_tier"] for item in TEST_PROMPTS]
_reference_embeddings = _model.encode(_reference_prompts)


def classify_embedding(prompt: str, k: int = 3) -> str:
    """
    Classifies a prompt's complexity by finding its k nearest neighbors
    (by semantic embedding similarity) among the labeled reference set,
    and returning the majority tier among them.
    """
    query_embedding = _model.encode([prompt])[0]

    # cosine similarity between the query and every reference prompt
    similarities = np.dot(_reference_embeddings, query_embedding) / (
        np.linalg.norm(_reference_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # get indices of the k most similar reference prompts
    top_k_indices = np.argsort(similarities)[-k:]
    top_k_tiers = [_reference_tiers[i] for i in top_k_indices]

    # majority vote among the k nearest neighbors
    return max(set(top_k_tiers), key=top_k_tiers.count)