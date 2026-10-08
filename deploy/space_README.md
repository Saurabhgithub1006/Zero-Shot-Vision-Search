---
title: Zero-Shot Driving Scenario Search
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Find driving scenarios in Audi A2D2 camera data by text
---

# Zero-Shot Driving Scenario Search

Describe a driving situation in plain language — *"a cyclist next to parked cars"*, *"traffic cones at a construction site"* — and get the matching frames from Audi's A2D2 test drives on German roads. No labelling, no retraining: SigLIP embeds text and images into the same vector space, and Pinecone retrieves the closest frames.

- **Use case:** scenario mining for ADAS validation (finding rare, safety-relevant situations in fleet data)
- **Quality:** mean Precision@10 = 0.79 vs 0.27 random base rate on 12 scenario queries, graded against A2D2's human labels
- **Filter:** restrict results to frames that contain specific classes (pedestrian, bicycle, zebra crossing, …)

Source code, evaluation and REST API: https://github.com/Saurabhgithub1006/Zero-Shot-Vision-Search

## Data & license

Images: **A2D2 – Audi Autonomous Driving Dataset**, © Audi AG, licensed under [CC BY-ND 4.0](https://creativecommons.org/licenses/by-nd/4.0/). Images are loaded unmodified from the official public A2D2 bucket and are not hosted in this Space. Geyer et al., *A2D2: Audi Autonomous Driving Dataset*, arXiv:2004.06320.
