import os
import sys
import csv
import random
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_loader import ModelLoader
from src.vector_indexer import Indexer
from src.ranker import Ranker

def evaluate(sample_size=100):
    # Configuration
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets/image-dataset')
    csv_path = os.path.join(os.path.dirname(__file__), '../assets/unsplash-research-dataset-lite-latest/photos.csv000')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print("Initializing components...")
    model_loader = ModelLoader()
    indexer = Indexer()
    print("Loading Ranker...")
    ranker = Ranker()
    
    print(f"Reading metadata from {csv_path}...")
    
    # Filter for available images
    print("Filtering for available images...")
    available_ids = set()
    if os.path.exists(assets_dir):
        for f in os.listdir(assets_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                photo_id = os.path.splitext(f)[0]
                available_ids.add(photo_id)
    
    if not available_ids:
        print("No images found in assets directory.")
        return

    # Read CSV and filter
    valid_rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row['photo_id'] in available_ids:
                    # Check if we have a description
                    desc = row.get('ai_description')
                    if not desc:
                        desc = row.get('photo_description')
                    
                    if desc:
                        row['used_description'] = desc
                        valid_rows.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if not valid_rows:
        print("No matching images found in CSV with descriptions.")
        return
        
    print(f"Found {len(valid_rows)} available images with descriptions for evaluation.")
    
    # Select random sample
    sample_size = min(len(valid_rows), sample_size)
    print(f"Selecting random sample of {sample_size} images...")
    sample_rows = random.sample(valid_rows, sample_size)
    
    # Metrics
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mrr_sum = 0
    
    # Build a lookup dictionary for descriptions
    # The initial read filtered for available images, so no need to read the CSV again.
    desc_lookup = {row['photo_id']: row['used_description'] for row in valid_rows}

    for row in tqdm(sample_rows):
        photo_id = row['photo_id']
        query_text = row['used_description']
            
        # Generate embedding for the query
        text_embedding = model_loader.get_text_embedding(query_text)
        
        if not text_embedding:
            continue
            
        # Search Pinecone (Fetch top 50 for re-ranking)
        results = indexer.search(text_embedding, top_k=100)
        
        candidates = []
        if results and results['matches']:
            for match in results['matches']:
                filename = match['metadata'].get('filename', '')
                pid = os.path.splitext(filename)[0]
                
                # Get description
                description = desc_lookup.get(pid, "")
                
                candidates.append({
                    'id': match['id'],
                    'text': description,
                    'metadata': match['metadata'],
                    'original_score': match['score']
                })
        
        # Re-rank
        ranked_results = ranker.rank(query_text, candidates, top_k=10)
        
        # Determine target ID
        filename = f"{photo_id}.jpg"
        if not os.path.exists(os.path.join(assets_dir, filename)):
             for f in os.listdir(assets_dir):
                 if f.startswith(photo_id):
                     filename = f
                     break
        
        rel_path = os.path.join('assets/image-dataset', filename)
        target_id = hashlib.md5(rel_path.encode()).hexdigest()
        
        rank = -1
        # Check in ranked_results
        for i, match in enumerate(ranked_results):
            if match['id'] == target_id:
                rank = i + 1
                break
        
        if rank > 0:
            mrr_sum += 1.0 / rank
            if rank <= 1:
                recall_at_1 += 1
            if rank <= 5:
                recall_at_5 += 1
            if rank <= 10:
                recall_at_10 += 1
                
    # Calculate final metrics
    print("\n--- Evaluation Results ---")
    print(f"Sample Size: {sample_size}")
    print(f"Recall@1:  {recall_at_1 / sample_size:.4f}")
    print(f"Recall@5:  {recall_at_5 / sample_size:.4f}")
    print(f"Recall@10: {recall_at_10 / sample_size:.4f}")
    print(f"MRR:       {mrr_sum / sample_size:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    evaluate()
