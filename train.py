"""Train the EOT classifier and save a single reusable model.pkl.

Usage:
    python train.py                      # pools english + hindi dev folders
    python train.py --data_dirs A B ...  # override folders

Reports honest grouped (by turn) cross-validation: AUC + the REAL scorer
delay on out-of-fold predictions (train never sees the turn it predicts),
then refits on all data and saves {scaler, clf, feature_names}.

The CV is REPEATED over several random fold assignments (--cv_seeds): a
single 5-fold split of ~100 turns/language has ±30-45 ms delay noise, which
is larger than most feature/model deltas — decisions are made on the
mean ± std across seeds, never on one split.

predict.py loads that model; it never refits.
"""
import argparse
import csv
import os

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from eot_features import extract_features, FEATURE_NAMES
from features import load_wav
import score as scorer

DEFAULT_DIRS = [
    "eot_handout/eot_data/english",
    "eot_handout/eot_data/hindi",
]


def build_dataset(data_dirs):
    X, y, groups, langs, keys, dir_of = [], [], [], [], [], []
    wav_cache = {}
    for d in data_dirs:
        lang = os.path.basename(d.rstrip("/"))
        with open(os.path.join(d, "labels.csv")) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            path = os.path.join(d, r["audio_file"])
            if path not in wav_cache:
                wav_cache[path] = load_wav(path)
            x, sr = wav_cache[path]
            feat = extract_features(x, sr, float(r["pause_start"]), int(r["pause_index"]))
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(f"{lang}:{r['turn_id']}")     # unique across languages
            langs.append(lang)
            keys.append((r["turn_id"], int(r["pause_index"])))
            dir_of.append(d)
    return (np.array(X), np.array(y), np.array(groups),
            np.array(langs), keys, dir_of)


def make_model():
    """Soft-voting ensemble: regularized linear model (carries the
    declination-to-floor / rhythm signal linearly and generalizes on ~500
    samples) + a bagged shallow gradient booster (nonlinear interactions,
    seed-averaged to kill single-seed variance). Chosen over a single
    HistGB, which memorized the training set (in-sample AUC 1.0) yet lost
    on out-of-fold AUC."""
    lin = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"))
    gbms = [(f"gbm{s}", GradientBoostingClassifier(
                n_estimators=150, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=s)) for s in range(3)]
    return VotingClassifier(estimators=[("lr", lin)] + gbms, voting="soft")


def random_group_folds(groups, seed, k=5):
    """k grouped folds from a seed-shuffled round-robin over unique turns."""
    rng = np.random.RandomState(seed)
    ug = np.array(sorted(set(groups)))
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    fid = np.array([fold_of[g] for g in groups])
    return [(np.where(fid != j)[0], np.where(fid == j)[0]) for j in range(k)]


def oof_score_per_language(oof_p, y, langs, keys, dir_of, tmpdir):
    """Write OOF predictions per language and run the real scorer on them."""
    os.makedirs(tmpdir, exist_ok=True)
    out = {}
    for d in sorted(set(dir_of)):
        lang = os.path.basename(d.rstrip("/"))
        idx = [i for i in range(len(dir_of)) if dir_of[i] == d]
        pred_csv = os.path.join(tmpdir, f"oof_{lang}.csv")
        with open(pred_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["turn_id", "pause_index", "p_eot"])
            for i in idx:
                w.writerow([keys[i][0], keys[i][1], f"{oof_p[i]:.6f}"])
        r = scorer.score(os.path.join(d, "labels.csv"), pred_csv)
        out[lang] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", default=DEFAULT_DIRS)
    ap.add_argument("--model_out", default="model.pkl")
    ap.add_argument("--tmpdir", default="scratch")
    ap.add_argument("--cv_seeds", type=int, default=5,
                    help="random grouped fold assignments to average over")
    args = ap.parse_args()

    X, y, groups, langs, keys, dir_of = build_dataset(args.data_dirs)
    print(f"dataset: {len(y)} pauses, {len(set(groups))} turns, "
          f"{int(y.sum())} eot / {int((1-y).sum())} hold, {X.shape[1]} features")

    # ---- repeated grouped CV: honest AUC + REAL scorer delay, mean over seeds ----
    seed_auc, seed_delay = [], {}
    for seed in range(args.cv_seeds):
        oof = np.zeros(len(y), dtype=float)
        for tr, te in random_group_folds(groups, seed):
            m = make_model()
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y, oof)
        seed_auc.append(auc)
        res = oof_score_per_language(oof, y, langs, keys, dir_of, args.tmpdir)
        line = "  ".join(f"{lang}={r['latency']*1000:.0f}ms(thr {r['threshold']}, "
                         f"delay {r['delay']*1000:.0f})" for lang, r in res.items())
        print(f"  seed {seed}: AUC={auc:.3f}  {line}")
        for lang, r in res.items():
            seed_delay.setdefault(lang, []).append(r["latency"] * 1000)

    print(f"grouped-CV over {args.cv_seeds} fold seeds "
          f"(mean±std — trust deltas only if they clear the std):")
    print(f"  OOF AUC : {np.mean(seed_auc):.3f} ± {np.std(seed_auc):.3f}")
    for lang, v in seed_delay.items():
        print(f"  {lang:8s}: {np.mean(v):6.0f} ± {np.std(v):.0f} ms")
    if len(seed_delay) == 2 and "hindi" in seed_delay:
        others = [l for l in seed_delay if l != "hindi"]
        sel = 0.75 * np.mean(seed_delay["hindi"]) + 0.25 * np.mean(seed_delay[others[0]])
        print(f"  selection metric (hidden set is mostly Hindi, 3:1): {sel:.0f} ms")

    # linear reference (sanity, seed-0 folds)
    lin = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    lin_oof = np.zeros(len(y))
    for tr, te in random_group_folds(groups, 0):
        lin.fit(X[tr], y[tr]); lin_oof[te] = lin.predict_proba(X[te])[:, 1]
    print(f"linear reference OOF AUC (seed 0): {roc_auc_score(y, lin_oof):.3f}")

    # ---- refit on ALL data and save ----
    model = make_model()
    model.fit(X, y)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, args.model_out)
    print(f"saved -> {args.model_out}")


if __name__ == "__main__":
    main()
