import os
import pandas as pd
import requests
from tqdm import tqdm
import argparse

def download_images(csv_path, output_dir, limit=10000, start_index=0):
    """
    Download images from Unsplash Lite dataset CSV.
    
    Args:
        csv_path (str): Path to the photos.csv000 file.
        output_dir (str): Directory to save images.
        limit (int): Maximum number of images to download.
        start_index (int): Index to start downloading from.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading {csv_path}...")
    try:
        # Read TSV file
        df = pd.read_csv(csv_path, sep='\t')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Slice the dataframe to start from start_index
    if start_index > 0:
        print(f"Starting from index {start_index}...")
        df = df.iloc[start_index:]

    print(f"Found {len(df)} images to process. Downloading top {limit}...")

    success_count = 0
    
    # Iterate through the dataframe
    for index, row in tqdm(df.iterrows(), total=min(len(df), limit)):
        if success_count >= limit:
            break
            
        photo_id = row['photo_id']
        photo_url = row['photo_image_url']
        
        # Construct output path
        # Using .jpg as default extension, though some might be png. 
        # Unsplash usually serves jpg.
        output_path = os.path.join(output_dir, f"{photo_id}.jpg")
        
        if os.path.exists(output_path):
            # print(f"Skipping {photo_id}, already exists.")
            success_count += 1
            continue

        try:
            response = requests.get(photo_url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
            else:
                pass
                # print(f"Failed to download {photo_id}: Status {response.status_code}")
        except Exception as e:
            pass
            # print(f"Error downloading {photo_id}: {e}")

    print(f"Download complete. Successfully downloaded {success_count} images to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from Unsplash Lite dataset.")
    parser.add_argument("--csv", type=str, default="assets/unsplash-research-dataset-lite-latest/photos.csv000", help="Path to the photos CSV file.")
    parser.add_argument("--output", type=str, default="assets/image-dataset", help="Directory to save images.")
    
    # Temporary defaults downloads the next n images  as requested by user
    #hardcoded for now as I have already donwloaded 11000 images and might want to download next n images to add more dataset.
    parser.add_argument("--limit", type=int, default=2000, help="Number of images to download.") 
    parser.add_argument("--start_index", type=int, default=11000, help="Index to start downloading from.")
    
    args = parser.parse_args()
    
    # temporary default to have start_index = 11000 and limit = 2000
    download_images(args.csv, args.output, args.limit, args.start_index)
