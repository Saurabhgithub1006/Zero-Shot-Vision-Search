# Publish the Streamlit app to a Hugging Face Space (Docker SDK).
# Uses your local `hf auth login` token. PINECONE_API_KEY must be added as a Space secret in the web UI.
import argparse
import os
import sys

from huggingface_hub import CommitOperationAdd, HfApi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import project_path

# Only what the container needs: no .env, no tests, no images
SPACE_FILES = [
    "Dockerfile",
    "requirements-space.txt",
    "app.py",
    "assets/a2d2/manifest.csv",
]
SPACE_DIRS = ["src"]


def files_to_upload():
    files = list(SPACE_FILES)
    for folder in SPACE_DIRS:
        for root, _, names in os.walk(project_path(folder)):
            if "__pycache__" in root:
                continue
            for name in names:
                if name.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, name), project_path())
                    files.append(rel.replace(os.sep, "/"))
    return files


def main(repo_id, dry_run):
    files = files_to_upload()
    missing = [f for f in files if not os.path.exists(project_path(f))]
    if missing:
        sys.exit(f"Missing files: {missing}")

    print(f"Space: {repo_id}")
    for f in files + ["README.md (from deploy/space_README.md)"]:
        print(f"  + {f}")
    if dry_run:
        return

    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    operations = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=project_path(f)) for f in files]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=project_path("deploy", "space_README.md")))

    api.create_commit(repo_id=repo_id, repo_type="space", operations=operations, commit_message="Deploy driving scenario search")

    print(f"\nDeployed: https://huggingface.co/spaces/{repo_id}")
    print("Next: Space Settings -> Variables and secrets -> New secret -> PINECONE_API_KEY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy the app to a Hugging Face Space.")
    parser.add_argument("--repo-id", required=True, help="e.g. your-hf-username/zero-shot-driving-search")
    parser.add_argument("--dry-run", action="store_true", help="List the files without uploading.")
    args = parser.parse_args()
    main(args.repo_id, args.dry_run)
