# Upload the local A2D2 subset (unmodified originals) to a public Hugging Face dataset repo,
# so the deployed app can load images from it. Resumable: re-run after an interruption.
import argparse
import csv
import os
import random
import sys

import requests
from huggingface_hub import HfApi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import project_path
from src.datasets.a2d2 import CLASS_LIST_PATH, DATA_DIR, MANIFEST_PATH, source_url

UPLOAD_PATTERNS = ["camera/**/*.png", "label/**/*.png", "manifest.csv", "class_list.json"]
CARD_PATH = project_path("deploy", "dataset_card.md")


def local_files():
    """Relative paths that will be uploaded, mirroring the repo layout."""
    files = []
    for sub in ("camera", "label"):
        for root, _, names in os.walk(os.path.join(DATA_DIR, sub)):
            files += [os.path.relpath(os.path.join(root, n), DATA_DIR).replace(os.sep, "/") for n in names if n.endswith(".png")]
    files += [os.path.basename(p) for p in (MANIFEST_PATH, CLASS_LIST_PATH) if os.path.exists(p)]
    return sorted(files)


def check_local(files):
    """Every manifest frame must have its camera image and label mask before uploading."""
    present = set(files)
    missing = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["image_path"].rsplit("/", 1)[-1]
            cam = f"camera/{row['scene']}/{name}"
            label = f"label/{row['scene']}/{name.replace('_camera_', '_label_')}"
            missing += [p for p in (cam, label) if p not in present]
    return missing


def verify_remote(api, repo_id, files, samples):
    """Compare remote and local file lists, then fetch sample images through the public URLs the app uses."""
    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    missing = [f for f in files if f not in remote]
    print(f"Remote files: {len(remote)} | local files missing remotely: {len(missing)}")
    if missing:
        print("  e.g.", missing[:5])

    cameras = [f for f in files if f.startswith("camera/")]
    ok = 0
    for path in random.sample(cameras, min(samples, len(cameras))):
        _, scene, name = path.split("/")
        resp = requests.get(source_url(scene, name, repo_id), timeout=60)
        ok += resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/")
    print(f"Public image URLs reachable: {ok}/{min(samples, len(cameras))}")
    return not missing and ok == min(samples, len(cameras))


def main(repo_id, dry_run, verify_only, samples):
    files = local_files()
    size_gb = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in files) / 1e9
    print(f"Dataset repo: {repo_id}")
    print(f"Local files: {len(files)} ({size_gb:.2f} GB) from {DATA_DIR}")

    missing = check_local(files)
    if missing:
        sys.exit(f"{len(missing)} files listed in the manifest are missing locally, e.g. {missing[:3]}. Run scripts/download_a2d2.py first.")

    api = HfApi()
    if verify_only:
        sys.exit(0 if verify_remote(api, repo_id, files, samples) else 1)
    if dry_run:
        print("Dry run: nothing uploaded.")
        return

    # Public on purpose: visitors' browsers load the images directly
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=CARD_PATH, path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                    commit_message="Add dataset card (CC BY-ND 4.0 attribution)")
    api.upload_large_folder(repo_id=repo_id, folder_path=DATA_DIR, repo_type="dataset", allow_patterns=UPLOAD_PATTERNS)

    print()
    verify_remote(api, repo_id, files, samples)
    print(f"\nDataset: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload the A2D2 subset to a Hugging Face dataset repo.")
    parser.add_argument("--repo-id", required=True, help="e.g. your-hf-username/a2d2-front-center-subset")
    parser.add_argument("--dry-run", action="store_true", help="Check local files without uploading.")
    parser.add_argument("--verify", action="store_true", help="Only check that the remote repo is complete and reachable.")
    parser.add_argument("--samples", type=int, default=10, help="Number of public image URLs to test.")
    args = parser.parse_args()
    main(args.repo_id, args.dry_run, args.verify, args.samples)
