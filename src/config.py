# Central settings: paths, model ids and index names. Secrets come from the environment only.
import os
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


@dataclass(frozen=True)
class Settings:
    dataset: str
    pinecone_index: str
    embedding_model: str = "google/siglip-so400m-patch14-384"
    embedding_dim: int = 1152
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = 50
    a2d2_hf_dataset: str = ""  # HF dataset repo hosting the frames; empty = Audi's public bucket


def get_settings():
    return Settings(
        dataset=os.environ.get("DATASET", "a2d2"),
        pinecone_index=os.environ.get("PINECONE_INDEX", "zero-shot-vision"),
        a2d2_hf_dataset=os.environ.get("A2D2_HF_DATASET", ""),
    )


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)
