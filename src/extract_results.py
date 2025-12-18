#!/usr/bin/env python3
import re
import csv
from pathlib import Path

GPU_RE = re.compile(r"^\s*GPU:\s*(.+?)\s*$", re.MULTILINE)
# Captures the number after the label (supports ints/floats and optional scientific notation)
SCORE_RE = re.compile(
    r"Overall leaderboard score\s*\(microseconds,\s*geom\)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)

def extract_gpu_and_score(text: str):
    gpu_match = GPU_RE.search(text)
    score_match = SCORE_RE.search(text)

    gpu = gpu_match.group(1).strip() if gpu_match else None
    score = float(score_match.group(1)) if score_match else None
    return gpu, score

def main():
    base = Path(".")
    folders = ["A100", "B200", "H100"]
    out_path = base / "gpu_scores.tsv"
    
    subfolder = 'batch'

    rows = []
    for folder in folders:
        d = base / "results" / folder / subfolder
        if not d.is_dir():
            continue

        for f in sorted(d.glob("test_*.out")):
            text = f.read_text(errors="replace")
            gpu, score = extract_gpu_and_score(text)
            if gpu is None or score is None:
                # Skip files that don't match; you can print a warning if you want.
                # print(f"Warning: missing fields in {f}")
                continue
            rows.append((gpu, score))

    # Write TSV
    with out_path.open("w", newline="") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow(["GPU", "Score"])
        for gpu, score in rows:
            w.writerow([gpu, score])

    print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == "__main__":
    main()
