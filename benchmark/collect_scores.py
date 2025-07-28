#!/usr/bin/env python3
import argparse
import json
import csv
from pathlib import Path

def get_score(data):
    # Try top-level (if any), then the usual nested path
    if "main_score" in data:
        return data["main_score"]
    try:
        return data["scores"]["test"][0]["main_score"]
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser(
        description="Extract nested main_score from JSONs into a one-column CSV"
    )
    p.add_argument("folder", type=Path, help="Folder with your JSON files")
    p.add_argument("-o", "--output", default="main_scores+avg.csv",
                   help="Output CSV filename")
    args = p.parse_args()

    folder = args.folder
    out_path = folder / args.output

    scores = []
    rows = []
    for jf in sorted(folder.glob("*.json")):
        if jf.name == "model_meta.json":
            continue
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            score = get_score(data)
        except Exception:
            score = None
        if score is not None:
            scores.append(score)
        rows.append([ "" if score is None else score ])

    # Compute average of non-None scores
    avg = sum(scores) / len(scores) if scores else None
    rows.append([avg, "AVG"])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["main_score", "note"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)+1} lines (including header) to {out_path}")

if __name__ == "__main__":
    main()
