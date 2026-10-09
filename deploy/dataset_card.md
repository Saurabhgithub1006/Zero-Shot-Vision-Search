---
license: cc-by-nd-4.0
pretty_name: A2D2 Front-Center Camera Subset (Semantic Segmentation)
task_categories:
- image-to-text
- image-segmentation
tags:
- autonomous-driving
- automotive
- adas
- a2d2
size_categories:
- 1K<n<10K
---

# A2D2 Front-Center Camera Subset

A subset of **1,978 front-center camera frames** with their **semantic segmentation label masks** from the semantic segmentation part of A2D2. Frames are sampled evenly from all 23 annotated scenes (recorded in southern Germany).

It is used by [Zero-Shot Driving Scenario Search](https://github.com/Saurabhgithub1006/Zero-Shot-Vision-Search) to display search results.

## Attribution and license

**A2D2 – Audi Autonomous Driving Dataset**, © Audi AG.
Licensed under [Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)](https://creativecommons.org/licenses/by-nd/4.0/).

- Original source: https://a2d2-dataset.github.io (public bucket `audi-autonomous-driving-dataset`, `camera_lidar_semantic/`)
- **No modifications:** every image and label mask is the unmodified original file. Only the selection of frames and the folder layout differ.
- Citation: Geyer et al., *A2D2: Audi Autonomous Driving Dataset*, arXiv:2004.06320, 2020.

## Structure

| Path | Content |
|---|---|
| `camera/<scene>/<scene-id>_camera_frontcenter_<frame>.png` | Original RGB frame (1920×1208) |
| `label/<scene>/<scene-id>_label_frontcenter_<frame>.png` | Original semantic label mask |
| `class_list.json` | Original A2D2 color → class mapping |
| `manifest.csv` | `scene`, `image_path`, `labels`: classes covering ≥ 1,500 pixels in each mask (numbered variants such as "Car 1"–"Car 4" are grouped as "Car"). This is project metadata derived from the masks. It is not part of A2D2 |
