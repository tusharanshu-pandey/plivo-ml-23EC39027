"""Inference for EOT detection. THE DELIVERABLE.

Runs, unmodified, on any folder with the same structure/labels schema:

    python predict.py --data_dir <folder> --out predictions.csv

Loads a pre-trained model (model.pkl) and writes:
    turn_id,pause_index,p_eot
for every pause in <folder>/labels.csv. It NEVER refits and NEVER reads
`label` or `pause_end` (see eot_features.py for the causality guarantee).
"""
import argparse
import csv
import os

import joblib
import numpy as np

from eot_features import extract_features, FEATURE_NAMES
from features import load_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl"))
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    assert bundle["feature_names"] == FEATURE_NAMES, "feature mismatch: retrain model.pkl"

    with open(os.path.join(args.data_dir, "labels.csv")) as f:
        rows = list(csv.DictReader(f))

    wav_cache, feats, keys = {}, [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in wav_cache:
            wav_cache[path] = load_wav(path)
        x, sr = wav_cache[path]
        feats.append(extract_features(x, sr, float(r["pause_start"]), int(r["pause_index"])))
        keys.append((r["turn_id"], r["pause_index"]))

    X = np.array(feats, dtype=np.float32)
    p = model.predict_proba(X)[:, 1] if len(X) else np.array([])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), prob in zip(keys, p):
            w.writerow([tid, pi, f"{float(prob):.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
