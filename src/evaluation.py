# Retrieval metrics and the A2D2 label-based query set.

# Scenario queries mapped to the A2D2 class that proves a result is relevant
A2D2_QUERIES = [
    ("a cyclist riding on the road", "Bicycle"),
    ("pedestrians walking near the street", "Pedestrian"),
    ("a truck on the road ahead", "Truck"),
    ("a traffic light at an intersection", "Traffic signal"),
    ("a road sign next to the street", "Traffic sign"),
    ("a zebra crossing on the road", "Zebra crossing"),
    ("a van or utility vehicle on the road", "Utility vehicle"),
    ("a motorcycle or scooter on the road", "Small vehicles"),
    ("a road with a cobblestone surface", "Drivable cobblestone"),
    ("a road with dashed lane markings", "Dashed line"),
    ("cars parked in a parking area", "Parking area"),
    ("traffic cones or barriers guiding cars", "Traffic guide obj."),
]


def rank_of(target_id, ranked_ids):
    """1-based position of target in the ranking, or None if absent."""
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == target_id:
            return i
    return None


def caption_metrics(ranks, sample_size, ks=(1, 5, 10)):
    """Recall@k and MRR for 'find the image described by its own caption' evaluation."""
    if sample_size == 0:
        return {}
    metrics = {f"Recall@{k}": sum(1 for r in ranks if r and r <= k) / sample_size for k in ks}
    metrics["MRR"] = sum(1.0 / r for r in ranks if r) / sample_size
    return metrics


def precision_at_k(result_labels, target_label, k):
    """Share of the top-k results whose ground truth contains the target class."""
    top = result_labels[:k]
    if not top:
        return 0.0
    return sum(1 for labels in top if target_label in labels) / len(top)


def base_rate(records, target_label):
    """Share of the whole dataset containing the class: what random retrieval would score."""
    if not records:
        return 0.0
    return sum(1 for r in records if target_label in r.labels) / len(records)
