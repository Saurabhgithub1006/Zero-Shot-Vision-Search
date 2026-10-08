# Zero-Shot Driving Scenario Search

Find safety-critical driving scenarios in automotive camera data by describing them in plain language — no labelling, no retraining.

> *"a cyclist next to parked cars"* → the matching frames from Audi's test drives on German roads, ranked by relevance.

## The problem

Automotive OEMs and suppliers record enormous volumes of fleet and test-drive camera data. Validating driver-assistance and automated-driving functions (e.g. for ISO 21448 / SOTIF scenario coverage) requires finding the rare situations hidden in that data: a cyclist between parked cars, a zebra crossing with pedestrians, a tractor on a country road. Manually tagging every frame is slow and expensive, and fixed-class detectors only find what they were trained for.

This project searches the data **zero-shot**: a vision-language model (SigLIP) embeds images and text into the same vector space, so any scenario that can be described in words can be searched, without training on it.

## Dataset

[A2D2 – Audi Autonomous Driving Dataset](https://a2d2-dataset.github.io), recorded by Audi on roads in southern Germany (city, rural roads, Autobahn).

- Front-center camera frames from all 23 annotated scenes (~2,000 frames, sampled evenly per scene)
- Pixel-level semantic labels are used as **ground truth** for evaluation and as optional search filters
- Downloaded directly from the official public AWS Open Data bucket (no account required)

**License:** A2D2 © Audi AG, licensed under [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/). Images are never committed to this repository and are displayed unmodified with attribution.

## Architecture

```mermaid
graph TD
    subgraph Offline
        DL[scripts/download_a2d2.py] -->|frames + label masks| DATA[assets/a2d2 + manifest.csv]
        DATA --> ING[scripts/ingest_and_index.py]
        ING -->|SigLIP image embeddings 1152-d| PC[(Pinecone · namespace per dataset)]
    end
    subgraph Online
        U[Engineer] --> APP[Streamlit app.py]
        T[Other tools] --> API[FastAPI api.py]
        APP --> SVC[src/search.py SearchService]
        API --> SVC
        SVC -->|text embedding| M[SigLIP so400m]
        SVC -->|top-k + label filter| PC
    end
    subgraph Evaluation
        EV[scripts/evaluate_model.py] --> SVC
        EV -->|Precision@10 vs label ground truth| R[results]
    end
```

| Module | Responsibility |
|---|---|
| `src/config.py` | Central settings (dataset, index, model ids); secrets only from the environment |
| `src/datasets/` | Dataset adapters (`a2d2`, `unsplash`) exposing images, captions and labels uniformly |
| `src/model_loader.py` | SigLIP image/text embeddings on CUDA / MPS / CPU |
| `src/vector_indexer.py` | Pinecone index with one namespace per dataset |
| `src/search.py` | Shared search pipeline used by the app, the API and evaluation |
| `src/evaluation.py` | Retrieval metrics and the scenario query set |
| `api.py` | REST API |
| `app.py` | Streamlit UI |

## Evaluation

Each scenario query is graded against A2D2's human-annotated label masks: a result counts as relevant when the target class is visible in that frame. **Precision@10** is compared with the **base rate** — the share of all frames containing the class, i.e. what random retrieval would score.

Results on 1,978 frames (`scripts/evaluate_model.py`, report in `artifacts/eval_a2d2.json`):

| Scenario query | Target class | Precision@10 | Base rate | Lift |
|---|---|---:|---:|---:|
| a cyclist riding on the road | Bicycle | 0.90 | 0.17 | 5.4× |
| pedestrians walking near the street | Pedestrian | 1.00 | 0.19 | 5.4× |
| a truck on the road ahead | Truck | 1.00 | 0.53 | 1.9× |
| a traffic light at an intersection | Traffic signal | 1.00 | 0.19 | 5.2× |
| a road sign next to the street | Traffic sign | 1.00 | 0.58 | 1.7× |
| a zebra crossing on the road | Zebra crossing | 0.30 | 0.01 | 54× |
| a van or utility vehicle on the road | Utility vehicle | 0.00 | 0.05 | 0× |
| a motorcycle or scooter on the road | Small vehicles | 0.70 | 0.04 | 20× |
| a road with a cobblestone surface | Drivable cobblestone | 1.00 | 0.39 | 2.6× |
| a road with dashed lane markings | Dashed line | 0.90 | 0.83 | 1.1× |
| cars parked in a parking area | Parking area | 0.80 | 0.15 | 5.4× |
| traffic cones or barriers guiding cars | Traffic guide obj. | 0.90 | 0.08 | 11× |
| **Mean** | | **0.79** | **0.27** | |

**Observations**
- Rare, safety-relevant scenarios benefit most: zebra crossings occur in 11 of 1,978 frames (0.6 %), and 3 of them appear in the top 10.
- Utility vehicles fail (0.00): the top 10 are frames with cars (9) and trucks (4), so the model finds vehicles but not this specific class. Hypothesis: the query wording doesn't match how SigLIP represents A2D2's "Utility vehicle" class; not yet verified. Combining the text query with the label filter (`labels: ["Utility vehicle"]`) returns 10 correct frames as a workaround.
- No re-ranking is applied for A2D2 because the dataset has no captions; the cross-encoder re-ranker is used automatically for captioned datasets (Unsplash).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/macOS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU build
pip install -r requirements.txt
copy .env.example .env            # then add your PINECONE_API_KEY to .env
```

## Usage

```bash
python scripts/download_a2d2.py --total 2000    # ~6 GB, resumable
python scripts/ingest_and_index.py              # embed + upsert (skips already indexed images)
python scripts/evaluate_model.py --output artifacts/eval_a2d2.json

streamlit run app.py                            # UI
uvicorn api:app --port 8000                     # REST API, docs at http://localhost:8000/docs
pytest                                          # tests
```

### REST API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status, active dataset and image count |
| GET | `/labels` | Ground-truth classes available for filtering |
| POST | `/search` | `{"query": "...", "top_k": 12, "labels": ["Bicycle"]}` → ranked results |
| GET | `/images/{id}` | Image file for a result |

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
     -d '{"query": "a cyclist next to parked cars", "top_k": 5}'
```

## Deploy to Hugging Face Spaces

The Space runs the Streamlit app on CPU in Docker (`Dockerfile`, `requirements-space.txt`). Images are loaded unmodified from the official A2D2 bucket, so the Space never hosts the dataset. Only the frame→class manifest (`assets/a2d2/manifest.csv`) ships with the code.

```bash
hf auth login                                                    # token with write access
python scripts/deploy_space.py --repo-id <hf-user>/zero-shot-driving-search --dry-run
python scripts/deploy_space.py --repo-id <hf-user>/zero-shot-driving-search
```

Then, in the Space: **Settings → Variables and secrets → New secret** → `PINECONE_API_KEY`. The index must already be populated (`scripts/ingest_and_index.py` run locally).

## Credits

- Based on [Tekraj15/zero-shot-vision-search](https://github.com/Tekraj15/zero-shot-vision-search) (original Unsplash-based implementation).
- Dataset: A2D2 © Audi AG, CC BY-ND 4.0 — Geyer et al., *A2D2: Audi Autonomous Driving Dataset*, arXiv:2004.06320.
- Model: [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384).
