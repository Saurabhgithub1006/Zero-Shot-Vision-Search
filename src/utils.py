import os
import json
import numpy as np
from PIL import Image

def load_metadata(metadata_path):
    """Load metadata from JSON file."""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata, metadata_path):
    """Save metadata to JSON file."""
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

def load_embeddings(embeddings_path):
    """Load embeddings from NPY file."""
    if os.path.exists(embeddings_path):
        return np.load(embeddings_path, allow_pickle=True)
    return None

def save_embeddings(embeddings, embeddings_path):
    """Save embeddings to NPY file."""
    np.save(embeddings_path, embeddings)

def get_image_paths(assets_dir):
    """
    Get all image paths from the assets directory.
    Supports .jpg, .jpeg, .png.
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    
    for root, _, files in os.walk(assets_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
                
    return image_paths
