# Changelog

## CHG-20261006-01 — Environment and repository hygiene — 2026-10-08

**What:** The project installs and runs on Python 3.13 with a CUDA 12.8 GPU build of torch, and secrets/data can no longer be committed by accident.

**How it works:** `.gitignore` excludes `.env`, key files, virtualenvs and all downloaded datasets. `.env.example` lists the required variables without values. `requirements.txt` replaces the uninstallable `torch==2.4.1` pin with `torch>=2.7` (RTX 50xx needs cu128 builds) and adds the API and test dependencies.

**Result:** `torch.cuda.is_available()` returns `True` on the RTX 5060. `.env` is confirmed git-ignored (`git check-ignore`).

**Files:** `.gitignore`, `.env.example`, `requirements.txt`

## CHG-20261006-02 — Shared modules and dataset adapters — 2026-10-08

**What:** Dataset paths, device selection, caption loading and the search pipeline each live in one place instead of being copied across scripts.

**How it works:** `src/config.py` holds settings, read from the environment. `src/device.py` selects CUDA, then MPS, then CPU. `src/datasets/` exposes every dataset as `ImageRecord`s (path, caption, labels) through one adapter interface. `src/search.py` is the single embed → Pinecone → optional re-rank pipeline, used by the app, the API and evaluation. Re-ranking switches on automatically only for captioned datasets. The Pinecone indexer uses one namespace per dataset.

**Result:** Switching datasets means setting `DATASET=` with no code edits. Unit tests cover every new module.

**Files:** `src/config.py`, `src/device.py`, `src/datasets/*`, `src/search.py`, `src/vector_indexer.py`, `src/ranker.py`, `src/utils.py`, `app.py`, `scripts/*`

## CHG-20261006-03 — Image ID and embedding compatibility fixes — 2026-10-08

**What:** Evaluation can match its targets on Windows, and embeddings work with transformers 5.x.

**How it works:** `src/ids.py` normalises every path to a project-relative POSIX form before hashing, so ingestion and evaluation produce the same vector ID on every OS (IDs built on POSIX paths are unchanged). `src/model_loader.py` takes `pooler_output` when transformers returns a model output object instead of a tensor.

**Result:** Regression tests cover both fixes. Embeddings were confirmed numerically identical to the model's own `image_embeds`/`text_embeds`.

**Files:** `src/ids.py`, `src/model_loader.py`, `tests/test_ids.py`, `tests/test_model_loader.py`

## CHG-20261006-04 — A2D2 dataset integration and indexing — 2026-10-08

**What:** The search engine runs on Audi's A2D2 driving dataset recorded on German roads.

**How it works:** `scripts/download_a2d2.py` lists the official public S3 bucket over HTTPS (no account needed) and downloads front-center frames plus their semantic label masks, spread evenly across all 23 scenes. Each mask is reduced to the set of classes covering at least 1,500 pixels, written to `manifest.csv` and stored as Pinecone metadata, where it serves as ground truth and as a search filter.

**Result:** 1,978 frames (6.8 GB) were downloaded and indexed. Pinecone reports 1,978 vectors. A second ingest run skips all 1,978.

**Files:** `scripts/download_a2d2.py`, `src/datasets/a2d2.py`, `scripts/ingest_and_index.py`

## CHG-20261006-05 — Scenario evaluation, REST API and app update — 2026-10-08

**What:** Retrieval quality is measured against human labels, and the search is exposed through a REST API and an updated Streamlit UI.

**How it works:** `scripts/evaluate_model.py` runs 12 scenario queries and scores Precision@10 against the label masks, next to each class's base rate. `api.py` (FastAPI) serves `/health`, `/labels`, `/search` (with an optional all-labels-required filter) and `/images/{id}` (lookup by ID only, so there's no path access). The app adds a class filter, label captions and the CC BY-ND attribution.

**Result:** Mean Precision@10 is 0.79 against a 0.27 base rate. In live HTTP tests every endpoint returned the expected data, and invalid input returned 422/404. The Streamlit AppTest rendered 12 results, and the filter returned only matching frames. 47 unit/API tests pass.

**Files:** `scripts/evaluate_model.py`, `src/evaluation.py`, `api.py`, `app.py`, `tests/*`, `artifacts/eval_a2d2.json`

## CHG-20261006-06 — Project documentation — 2026-10-08

**What:** The README describes the scenario-search problem, the dataset license, the architecture, the measured results and how to run it, with credit to the upstream project.

**How it works:** Documentation only.

**Result:** README, roadmap status, changelog and audit log are up to date.

**Files:** `README.md`, `artifacts/system-arch-and-roadmap.md`, `artifacts/changelogs.md`, `artifacts/audit.md`
