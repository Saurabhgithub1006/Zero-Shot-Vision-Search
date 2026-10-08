# REST API for scenario search, so other tools (labeling, test pipelines) can query the index.
# Run: uvicorn api:app --port 8000
import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

from src.config import project_path
from src.search import SearchService, build_search_service

app = FastAPI(
    title="Zero-Shot Vision Search API",
    description="Natural-language search over driving scenes (A2D2, CC BY-ND 4.0, Audi AG).",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=300, examples=["a cyclist next to parked cars"])
    top_k: int = Field(12, ge=1, le=50)
    labels: List[str] = Field(default_factory=list, description="Only return images containing all of these classes.")


class SearchHit(BaseModel):
    id: str
    filename: str
    score: float
    vector_score: float
    labels: List[str]
    scene: str
    caption: str
    image_url: str


class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchHit]


@lru_cache(maxsize=1)
def get_service() -> SearchService:
    return build_search_service()


@app.get("/health")
def health(service: SearchService = Depends(get_service)):
    return {"status": "ok", "dataset": service.dataset.name, "images": len(service.records)}


@app.get("/labels", response_model=List[str])
def labels(service: SearchService = Depends(get_service)):
    return sorted({label for record in service.records.values() for label in record.labels})


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, service: SearchService = Depends(get_service)):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be blank.")
    try:
        results = service.search(request.query, top_k=request.top_k, labels=request.labels)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    hits = [
        SearchHit(
            id=r.id, filename=r.filename, score=r.score, vector_score=r.vector_score,
            labels=r.labels, scene=r.scene, caption=r.caption, image_url=f"/images/{r.id}",
        )
        for r in results
    ]
    return SearchResponse(query=request.query, count=len(hits), results=hits)


@app.get("/images/{image_id}")
def image(image_id: str, service: SearchService = Depends(get_service)):
    # Only ids known to the dataset are served, so arbitrary paths can't be requested
    record = service.records.get(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown image id.")
    path = project_path(record.path)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image file not found on disk.")
    return FileResponse(path)
