import argparse
import json
import os
import random
import sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import project_path
from src.evaluation import (
    A2D2_QUERIES, base_rate, caption_metrics, precision_at_k, rank_of,
)
from src.search import build_search_service


def evaluate_captions(service, sample_size, seed):
    """Each image's own caption is the query; the image itself is the only correct answer."""
    captioned = [r for r in service.dataset.records() if r.caption]
    if not captioned:
        print("No captioned images found.")
        return {}

    random.seed(seed)
    sample = random.sample(captioned, min(sample_size, len(captioned)))
    ranks = []
    for record in tqdm(sample, desc="Queries"):
        results = service.search(record.caption, top_k=10)
        ranks.append(rank_of(record.id, [r.id for r in results]))
    return {"sample_size": len(sample), **caption_metrics(ranks, len(sample))}


def evaluate_labels(service, k):
    """Scenario queries graded by the ground-truth classes in the label masks."""
    records = service.dataset.records()
    rows = []
    for query, target in tqdm(A2D2_QUERIES, desc="Queries"):
        results = service.search(query, top_k=k)
        rows.append({
            "query": query,
            "target_class": target,
            f"precision@{k}": precision_at_k([r.labels for r in results], target, k),
            "base_rate": base_rate(records, target),
        })
    return {"images": len(records), "k": k, "queries": rows}


def print_report(report):
    print("\n--- Evaluation Results ---")
    if "queries" in report:
        k = report["k"]
        print(f"Images: {report['images']}  |  metric: Precision@{k} vs random-retrieval base rate\n")
        print(f"{'Query':45} {'Class':22} {'P@' + str(k):>6} {'Base':>6}")
        for row in report["queries"]:
            print(f"{row['query']:45} {row['target_class']:22} {row[f'precision@{k}']:6.2f} {row['base_rate']:6.2f}")
        mean_p = sum(r[f"precision@{k}"] for r in report["queries"]) / len(report["queries"])
        mean_b = sum(r["base_rate"] for r in report["queries"]) / len(report["queries"])
        print(f"\n{'Mean':68} {mean_p:6.2f} {mean_b:6.2f}")
    else:
        for name, value in report.items():
            print(f"{name:12} {value:.4f}" if isinstance(value, float) else f"{name:12} {value}")
    print("--------------------------")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on the configured dataset.")
    parser.add_argument("--sample_size", type=int, default=100, help="Caption mode: number of sampled queries.")
    parser.add_argument("--k", type=int, default=10, help="Label mode: precision cut-off.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file for the report.")
    args = parser.parse_args()

    service = build_search_service()
    if service.dataset.has_labels:
        report = evaluate_labels(service, args.k)
    else:
        report = evaluate_captions(service, args.sample_size, args.seed)

    print_report(report)
    if args.output:
        out = project_path(args.output) if not os.path.isabs(args.output) else args.output
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
