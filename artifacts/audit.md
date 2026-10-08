# Audit Log

## CHG-20261006-01 — 2026-10-08

**Issue:** The cloned project could not be installed on the target machine, and nothing prevented secrets or datasets from being committed.

**Root cause:** `requirements.txt:1` pinned `torch==2.4.1`, which has no Python 3.13 wheels (`pip index versions torch` lists 2.6.0 as the oldest), and the RTX 5060 needs cu128 builds. The repository had no `.gitignore`.

**Impact:** The project could not be installed at all. There was a risk of committing `PINECONE_API_KEY`.

**Fix implemented:** Unpinned modern GPU torch and git-ignored secrets and datasets

**Decisions made:**
- D2: Python 3.13 venv with torch ≥ 2.7 (cu128). Recommended by the agent, approved by the user.

**Alternatives rejected:** A Python 3.12 venv (needs a second interpreter install for no benefit).
**Validation:** `torch.cuda.is_available()` returns `True`. `git check-ignore .env` matches.
**Follow-ups:** None.

## CHG-20261006-02 — 2026-10-08

**Issue:** Retargeting to a new dataset needed edits in many files.

**Root cause:** Dataset paths were hardcoded in `app.py`, `scripts/evaluate_model.py` and `scripts/download_images.py`. Device selection was copied in `src/model_loader.py` and `src/ranker.py`. Caption loading was copied in `app.py` and `scripts/evaluate_model.py`.

**Impact:** No quantifiable metric available (maintainability).

**Fix implemented:** Centralised configuration with pluggable dataset adapters and a single search service

**Decisions made:**
- D3: One Pinecone index (`zero-shot-vision`) with a namespace per dataset. Recommended by the agent, approved.
- D4: Unsplash files kept; the Unsplash adapter stays usable.

**Alternatives rejected:** One index per dataset (uses up free-tier index slots). Local FAISS (differs from upstream; larger change).
**Validation:** Unit tests for config, device, datasets and search (47 tests in total pass).
**Follow-ups:** None.

## CHG-20261006-03 — 2026-10-08

**Issue:** (a) Evaluation would report 0 for every metric on Windows. (b) Image and text embedding failed for every input.

**Root cause:** (a) `scripts/ingest_and_index.py` hashed `os.path.relpath` (backslashes) while `scripts/evaluate_model.py` hashed `os.path.join('assets/image-dataset', f)` (mixed separators), so the IDs never matched (reproduced: the hashes differ). (b) In transformers 5.x, `get_image_features`/`get_text_features` return `BaseModelOutputWithPooling`, and `src/model_loader.py` called `.norm()` on it.

**Impact:** (a) 100% of evaluation targets were unmatchable on Windows. (b) 100% of embeddings failed (6/6 in the first smoke test), which would have produced an empty index.

**Fix implemented:** OS-independent vector IDs and transformers-version-agnostic feature extraction

**Decisions made:** None (minimal fixes, behaviour unchanged on POSIX and on transformers 4.x).

**Alternatives rejected:** Pinning transformers < 5 (blocks upgrades and still leaves a latent bug).
**Validation:** Regression tests. Embeddings equal the forward-pass `image_embeds`/`text_embeds` (`np.allclose`, atol 1e-4).
**Follow-ups:** None.

## CHG-20261006-04 — 2026-10-08

**Issue:** The user asked for a legally usable automotive dataset relevant to German industry.

**Root cause:** Not a defect (new feature).

**Impact:** 1,978 frames from 23 scenes indexed. Download 6.8 GB.

**Fix implemented:** A2D2 front-camera subset with ground truth derived from the label masks

**Decisions made:**
- D1: A2D2. Chosen by the user after a license review (CC BY-ND 4.0, commercial use allowed, public bucket without registration).

**Alternatives rejected:** Cityscapes (registration needs a scientific affiliation). CoVLA (academic/non-commercial only, Japan, 452 GB). CarDD (terms unverified, form required). OSDaR23 (license could not be verified).
**Validation:** Pinecone `describe_index_stats` reports 1,978 vectors. An idempotent re-run reports "Processed: 0, Skipped: 1978".
**Follow-ups:** Images must never be committed or redistributed in modified form (ND clause).

## CHG-20261006-05 — 2026-10-08

**Issue:** The new dataset has no captions, so the upstream caption-based evaluation and re-ranking could not be used. The user requested API endpoints with validation.

**Root cause:** The upstream design assumed a caption for every image (`app.py:101`, `src/ranker.py:38`).

**Impact:** Mean Precision@10 0.79 vs a 0.27 random base rate across 12 scenario queries. Search latency is about 1 s per API call, including the Pinecone round-trip.

**Fix implemented:** Label-graded scenario benchmark with a REST search service and filterable UI

**Decisions made:**
- Re-ranker gated to captioned datasets only. Comparing queries to empty captions would make ranking arbitrary.
- The "Tractor" query was replaced by "Traffic guide obj." because Tractor occurs in only 1 frame (Precision@10 could never exceed 0.1).
- The REST API was added because the user asked for API endpoint testing.

**Alternatives rejected:** Generating captions with a VLM now (deferred; it would grade a model with model-made labels).
**Validation:** 47 tests pass. Live HTTP: `/health` 200, `/labels` 200, `/search` returned correct classes, label filter 5/5 matching, `/images/{id}` 200 PNG; empty query, `top_k` 0/99, `{}` and non-JSON → 422; unknown ID and traversal → 404. Streamlit AppTest: no exceptions, 12 images, filter returned 11/11 zebra-crossing frames.
**Follow-ups:** "Utility vehicle" query scores 0.00 (cause unverified). Silence the `use_fast` deprecation warning.

## CHG-20261006-06 — 2026-10-08

**Issue:** The README described the upstream Unsplash project.

**Root cause:** Not a defect (documentation).

**Impact:** No quantifiable metric available.

**Fix implemented:** Project documentation for driving-scenario search with license attribution

**Decisions made:**
- D5: Credit line to the upstream repository. Recommended by the agent, approved.

**Alternatives rejected:** None.
**Validation:** README commands match the scripts that were run during validation.
**Follow-ups:** None.
