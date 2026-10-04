import os
import sys
import uuid
from tqdm import tqdm
from dotenv import load_dotenv
import hashlib

load_dotenv()

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_loader import ModelLoader
from src.vector_indexer import Indexer
from src.utils import get_image_paths, save_metadata, save_embeddings

def main():
    # Configuration
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets/image-dataset')
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    metadata_path = os.path.join(data_dir, 'metadata.json')
    embeddings_path = os.path.join(data_dir, 'embeddings.npy')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    model_loader = ModelLoader()
    indexer = Indexer()
    
    # Get image paths
    print(f"Scanning {assets_dir} for images...")
    image_paths = get_image_paths(assets_dir)
    if not image_paths:
        print(f"No images found in {assets_dir}. Please add images.")
        return
    
    print(f"Found {len(image_paths)} images.")
    
    # Process images
    metadata = {}
    batch_size = 100
    current_batch = []
    
    print(f"Generating embeddings and upserting in batches of {batch_size}...")
    
    # Pre-calculate Deterministic IDs and check existence in chunks
    total_skipped = 0
    total_processed = 0
    
    # Process in chunks of batch_size
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # 1. Generate Deterministic IDs for this batch, based on relative path
        batch_ids = []
        path_map = {} # Map ID to (path, rel_path)
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        for img_path in batch_paths:
            rel_path = os.path.relpath(img_path, project_root)
            
            img_id = hashlib.md5(rel_path.encode()).hexdigest()
            batch_ids.append(img_id)
            path_map[img_id] = (img_path, rel_path)
            
        # 2. Check which IDs already exist in Pinecone
        existing_vectors = indexer.fetch_vectors(batch_ids)
        existing_ids = set(existing_vectors.get('vectors', {}).keys())
        
        # 3. Filter out existing ones
        to_process_ids = [bid for bid in batch_ids if bid not in existing_ids]
        skipped_count = len(batch_ids) - len(to_process_ids)
        total_skipped += skipped_count
        
        if skipped_count > 0:
            print(f"Skipping {skipped_count} images already in index...")
            
        if not to_process_ids:
            continue
            
        # 4. Process only the missing ones
        current_batch = []
        for img_id in to_process_ids:
            img_path, rel_path = path_map[img_id]
            
            # Get embedding
            embedding = model_loader.get_image_embedding(img_path)
            
            if embedding:
                meta = {
                    "path": rel_path,
                    "filename": os.path.basename(img_path)
                }
                current_batch.append((img_id, embedding, meta))
                metadata[img_id] = meta
                total_processed += 1
        
        # 5. Upsert the new vectors
        if current_batch:
            indexer.upsert_vectors(current_batch)

    print(f"Processing complete. Processed: {total_processed}, Skipped: {total_skipped}")
    
    # Save local data (merge with existing if possible, but for now just saving what we processed)
    if os.path.exists(metadata_path):
        import json
        try:
            with open(metadata_path, 'r') as f:
                existing_metadata = json.load(f)
                existing_metadata.update(metadata)
                metadata = existing_metadata
        except Exception as e:
            print(f"Warning: Could not merge with existing metadata: {e}")
            
    print("Saving local metadata...")
    save_metadata(metadata, metadata_path)
    print("Ingestion complete!")


if __name__ == "__main__":
    main()
