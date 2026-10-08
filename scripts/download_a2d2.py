# Download a subset of A2D2 front-center frames + label masks from the official public bucket,
# then build manifest.csv with the ground-truth classes visible in each frame.
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.a2d2 import (
    CLASS_LIST_PATH, DATA_DIR, MANIFEST_PATH,
    label_classes_from_file, load_color_map, write_manifest,
)
from src.ids import to_rel_posix

BUCKET_URL = "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com"
ROOT_PREFIX = "camera_lidar_semantic/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_s3(prefix, delimiter=None):
    """Yield (keys, common_prefixes) pages for a prefix, following continuation tokens."""
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        resp = requests.get(BUCKET_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        keys = [k.text for k in root.findall("s3:Contents/s3:Key", S3_NS)]
        prefixes = [p.text for p in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS)]
        yield keys, prefixes
        if root.findtext("s3:IsTruncated", namespaces=S3_NS) != "true":
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=S3_NS)


def list_scenes():
    return [p for _, prefixes in list_s3(ROOT_PREFIX, delimiter="/") for p in prefixes]


def list_frames(scene_prefix):
    keys = [k for page, _ in list_s3(scene_prefix + "camera/cam_front_center/") for k in page]
    return sorted(k for k in keys if k.endswith(".png"))


def evenly_spaced(items, n):
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def download(key, dest):
    if os.path.exists(dest):
        return
    resp = requests.get(f"{BUCKET_URL}/{key}", timeout=60)
    resp.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(resp.content)
    os.replace(tmp, dest)


def label_key_for(camera_key):
    return camera_key.replace("/camera/", "/label/").replace("_camera_", "_label_")


def main(total, workers):
    os.makedirs(DATA_DIR, exist_ok=True)
    download(ROOT_PREFIX + "class_list.json", CLASS_LIST_PATH)

    scenes = list_scenes()
    per_scene = max(1, total // len(scenes))
    print(f"{len(scenes)} scenes, taking {per_scene} frames from each")

    # Pick frames evenly within every scene so the subset covers all routes and conditions
    jobs = []
    for scene_prefix in tqdm(scenes, desc="Listing scenes"):
        scene = scene_prefix.rstrip("/").rsplit("/", 1)[-1]
        for cam_key in evenly_spaced(list_frames(scene_prefix), per_scene):
            name = cam_key.rsplit("/", 1)[-1]
            cam_dest = os.path.join(DATA_DIR, "camera", scene, name)
            label_dest = os.path.join(DATA_DIR, "label", scene, name.replace("_camera_", "_label_"))
            jobs.append((scene, cam_key, cam_dest, label_key_for(cam_key), label_dest))

    for _, _, cam_dest, _, label_dest in jobs:
        os.makedirs(os.path.dirname(cam_dest), exist_ok=True)
        os.makedirs(os.path.dirname(label_dest), exist_ok=True)

    def fetch(job):
        _, cam_key, cam_dest, label_key, label_dest = job
        download(cam_key, cam_dest)
        download(label_key, label_dest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(fetch, jobs), total=len(jobs), desc="Downloading frames"))

    # Ground truth: which classes are visible in each frame
    color_map = load_color_map()
    rows = []
    for scene, _, cam_dest, _, label_dest in tqdm(jobs, desc="Reading label masks"):
        rows.append({
            "scene": scene,
            "image_path": to_rel_posix(cam_dest),
            "labels": label_classes_from_file(label_dest, color_map),
        })
    write_manifest(rows)
    print(f"Done: {len(rows)} frames, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an A2D2 front-camera subset (CC BY-ND 4.0).")
    parser.add_argument("--total", type=int, default=2000, help="Approximate number of frames to download.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads.")
    args = parser.parse_args()
    main(args.total, args.workers)
