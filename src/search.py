# Shared search pipeline used by the Streamlit app, the REST API and the evaluation script:
# text -> SigLIP embedding -> Pinecone top-k -> optional caption re-ranking -> results.
from dataclasses import dataclass, field

from src.utils import get_field


@dataclass
class SearchResult:
    id: str
    path: str
    filename: str
    score: float
    vector_score: float
    labels: list = field(default_factory=list)
    caption: str = ""
    scene: str = ""
    source_url: str = ""  # public original, used when the image isn't stored locally


def label_filter(labels):
    """Pinecone filter requiring every given label to be present."""
    if not labels:
        return None
    return {"$and": [{"labels": {"$in": [label]}} for label in labels]}


class SearchService:
    def __init__(self, embedder, indexer, dataset, ranker=None, candidate_k=50):
        self.embedder = embedder
        self.indexer = indexer
        self.dataset = dataset
        self.candidate_k = candidate_k
        # Re-ranking compares the query with image captions, so it only applies to captioned datasets
        self.ranker = ranker if dataset.has_captions else None
        self._records = None

    @property
    def records(self):
        if self._records is None:
            self._records = self.dataset.by_id()
        return self._records

    def search(self, query, top_k=12, labels=None):
        query = (query or "").strip()
        if not query:
            raise ValueError("Query must not be empty.")

        vector = self.embedder.get_text_embedding(query)
        if vector is None:
            raise RuntimeError("Failed to embed the query.")

        fetch_k = max(top_k, self.candidate_k) if self.ranker else top_k
        response = self.indexer.search(vector, top_k=fetch_k, metadata_filter=label_filter(labels))
        results = [self._to_result(m) for m in get_field(response, "matches", []) or []]

        if self.ranker and results:
            results = self._rerank(query, results, top_k)
        return results[:top_k]

    def _to_result(self, match):
        meta = dict(get_field(match, "metadata", {}) or {})
        match_id = get_field(match, "id")
        record = self.records.get(match_id)
        score = float(get_field(match, "score", 0.0))
        return SearchResult(
            id=match_id,
            path=meta.get("path", record.path if record else ""),
            filename=meta.get("filename", record.filename if record else ""),
            score=score,
            vector_score=score,
            labels=list(meta.get("labels", record.labels if record else [])),
            caption=(record.caption if record else None) or "",
            scene=meta.get("scene", ""),
            source_url=(record.extra.get("source_url", "") if record else ""),
        )

    def _rerank(self, query, results, top_k):
        candidates = [{"text": r.caption, "result": r} for r in results]
        ranked = self.ranker.rank(query, candidates, top_k=top_k)
        reranked = []
        for cand in ranked:
            cand["result"].score = cand["score"]
            reranked.append(cand["result"])
        return reranked


def build_search_service():
    """Wire the production components from config (loads models and connects to Pinecone)."""
    from src.config import get_settings
    from src.datasets import get_dataset
    from src.model_loader import ModelLoader
    from src.vector_indexer import Indexer

    settings = get_settings()
    dataset = get_dataset(settings.dataset)
    ranker = None
    if dataset.has_captions:
        from src.ranker import Ranker
        ranker = Ranker(settings.reranker_model)
    return SearchService(ModelLoader(), Indexer(), dataset, ranker=ranker, candidate_k=settings.candidate_k)
