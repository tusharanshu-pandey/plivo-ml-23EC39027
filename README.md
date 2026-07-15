# plivo-ml-23EC39027 — End-of-Turn Detection

Predict, for every pause in a user turn, `p_eot` (probability the turn is over), from **causal**
prosodic features (audio strictly before the pause). CPU-only, no pretrained models.

## Results (honest, grouped-by-turn out-of-fold)

| | English | Hindi |
|---|---|---|
| Silence-only baseline | 1600 ms | 850 ms |
| **This model** | **1209 ms** | **850 ms** |

Mean response delay at ≤5% interrupted turns (lower is better). Overall OOF AUC **0.685**.
English improves ~24%; Hindi is at its acoustic ceiling (see `NOTES.md` / `RUNLOG.md`).

## Files
- `eot_features.py` — 30 causal prosodic features (single source of truth; causality asserted in-code).
- `train.py` — pools EN+HI, grouped-CV (honest AUC + OOF scorer delay), saves `model.pkl`.
- `predict.py` — **the deliverable**; loads `model.pkl`, runs on any folder.
- `model.pkl` — trained ensemble (LogisticRegression + bagged GradientBoosting).
- `predictions_{english,hindi}.csv` — predictions for both provided folders.
- `score.py` — official scorer (unchanged). `features.py` — provided audio utilities.
- `RUNLOG.md`, `NOTES.md`, `SUMMARY.html` — write-ups. `tools/` — error-listening + summary builders.

## Reproduce
```bash
source ~/speedrun/env/bin/activate        # numpy scipy scikit-learn pandas librosa soundfile

# train (writes model.pkl, prints grouped-CV AUC + out-of-fold delay per language)
python train.py

# predict on any folder with the same labels.csv schema
python predict.py --data_dir eot_handout/eot_data/hindi --out predictions_hindi.csv

# score
python score.py --data_dir eot_handout/eot_data/hindi --pred predictions_hindi.csv
```
(`eot_handout/eot_data/*/audio/*.wav` is git-ignored; the grader runs `predict.py` on their own folder.)
