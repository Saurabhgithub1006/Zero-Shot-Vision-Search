import os
import sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_settings, project_path
from src.datasets import get_dataset
from src.model_loader import ModelLoader
from src.vector_indexer import Indexer
from src.utils import get_field, load_metadata, save_metadata


def build_metadata(record):
    """Metadata stored next to each vector (Pinecone only accepts flat values / string lists)."""
    meta = {"path": record.path, "filename": record.filename}
    if record.labels:
        meta["labels"] = record.labels
    if record.extra.get("scene"):
        meta["scene"] = record.extra["scene"]
    return meta


def main(batch_size=100):
    settings = get_settings()
    dataset = get_dataset(settings.dataset)
    data_dir = project_path('data')
    metadata_path = os.path.join(data_dir, f'metadata_{dataset.name}.json')
    os.makedirs(data_dir, exist_ok=True)

    records = dataset.records()
    if not records:
        print(f"No images found for dataset '{dataset.name}'. Download it first.")
        return
    print(f"Dataset '{dataset.name}': {len(records)} images.")

    print("Initializing components...")
    model_loader = ModelLoader()
    indexer = Indexer()

    metadata = {}
    total_processed = 0
    total_skipped = 0

    for i in tqdm(range(0, len(records), batch_size), desc="Batches"):
        batch = {r.id: r for r in records[i:i + batch_size]}

        # Skip images that are already in the index (re-runs are cheap)
        existing = indexer.fetch_vectors(list(batch))
        existing_ids = set(get_field(existing, 'vectors') or {})
        total_skipped += len(existing_ids)

        vectors = []
        for img_id, record in batch.items():
            if img_id in existing_ids:
                continue
            embedding = model_loader.get_image_embedding(os.path.join(project_path(), record.path))
            if embedding:
                meta = build_metadata(record)
                vectors.append((img_id, embedding, meta))
                metadata[img_id] = meta
                total_processed += 1

        if vectors:
            indexer.upsert_vectors(vectors)

    print(f"Processing complete. Processed: {total_processed}, Skipped: {total_skipped}")

    existing_metadata = load_metadata(metadata_path)
    existing_metadata.update(metadata)
    save_metadata(existing_metadata, metadata_path)
    print("Ingestion complete!")


if __name__ == "__main__":
    main()
