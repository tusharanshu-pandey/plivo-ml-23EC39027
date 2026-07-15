"""Dump the worst out-of-fold errors as short pre-pause audio clips + a table,
so a human can LISTEN and diagnose what the features miss.

    python tools/dump_errors.py [--n 12] [--clip_s 4] [--out_dir scratch/errors]

Two error types (ranked by how badly the model was wrong):
  - FALSE_CUTOFF : a HOLD pause the model scored HIGH (would interrupt the user)
  - MISS         : an EOT pause the model scored LOW (agent keeps waiting -> 1.6 s)

Clips are audio[max(0, pause_start-clip_s) : pause_start] -- strictly the
speech BEFORE the pause (what a live agent would have heard). Listen: does it
SOUND finished (falling pitch, energy fading, final lengthening) or mid-thought
(level/rising pitch, abrupt cut, filler)?
"""
import argparse
import csv
import os
import sys

import numpy as np
import soundfile as sf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import build_dataset, make_model, DEFAULT_DIRS
from features import load_wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", default=DEFAULT_DIRS)
    ap.add_argument("--n", type=int, default=12, help="clips per error type")
    ap.add_argument("--clip_s", type=float, default=4.0)
    ap.add_argument("--out_dir", default="scratch/errors")
    ap.add_argument("--only_lang", default=None, help="restrict shown clips to this language")
    args = ap.parse_args()

    X, y, groups, langs, keys, dir_of = build_dataset(args.data_dirs)

    # honest out-of-fold probabilities
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_model(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    print(f"OOF AUC={roc_auc_score(y, oof):.3f}  ({len(y)} pauses)")

    # need pause timings for clip extraction: reload labels
    meta = {}
    for d in args.data_dirs:
        with open(os.path.join(d, "labels.csv")) as f:
            for r in csv.DictReader(f):
                lang = os.path.basename(d.rstrip("/"))
                meta[(lang, r["turn_id"], int(r["pause_index"]))] = (
                    d, r["audio_file"], float(r["pause_start"]), float(r["pause_end"]))

    rows = []
    for i in range(len(y)):
        lang = langs[i]; tid, pi = keys[i]
        d, af, ps, pe = meta[(lang, tid, pi)]
        rows.append(dict(i=i, lang=lang, turn_id=tid, pause_index=pi, p=oof[i],
                         label="eot" if y[i] else "hold",
                         pause_start=ps, pause_end=pe, dur=pe - ps,
                         data_dir=d, audio_file=af))

    pool = [r for r in rows if (args.only_lang is None or r["lang"] == args.only_lang)]
    false_cut = sorted([r for r in pool if r["label"] == "hold"], key=lambda r: -r["p"])[:args.n]
    misses = sorted([r for r in pool if r["label"] == "eot"], key=lambda r: r["p"])[:args.n]

    os.makedirs(args.out_dir, exist_ok=True)
    table = []
    for kind, group in [("FALSE_CUTOFF", false_cut), ("MISS", misses)]:
        for rank, r in enumerate(group, 1):
            x, sr = load_wav(os.path.join(r["data_dir"], r["audio_file"]))
            end = int(r["pause_start"] * sr)
            start = max(0, end - int(args.clip_s * sr))
            clip = x[start:end]
            name = f"{kind}_{rank:02d}_{r['lang']}_{r['turn_id']}_p{r['p']:.2f}_dur{r['dur']:.1f}.wav"
            sf.write(os.path.join(args.out_dir, name), clip, sr)
            table.append({"clip": name, "kind": kind, "lang": r["lang"],
                          "turn_id": r["turn_id"], "pause_index": r["pause_index"],
                          "p_eot": round(r["p"], 3), "true": r["label"],
                          "pause_start_s": round(r["pause_start"], 2),
                          "hold_dur_s": round(r["dur"], 2)})

    with open(os.path.join(args.out_dir, "errors.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)

    print(f"\nwrote {len(table)} clips + errors.csv -> {args.out_dir}/\n")
    print(f"{'clip':52s} {'p_eot':>6s} {'true':>5s} {'holddur':>7s}")
    for t in table:
        print(f"{t['clip']:52s} {t['p_eot']:6.3f} {t['true']:>5s} {t['hold_dur_s']:7.2f}")


if __name__ == "__main__":
    main()
