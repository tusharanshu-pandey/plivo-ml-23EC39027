# Plan — End-of-Turn (EOT) Detection · Plivo Assignment

**Candidate:** Tusharanshu Pandey (23EC39027) · **Submission repo:** `plivo-ml-23EC39027` (public)
**Track chosen:** STT / End-of-Turn Detection · **Compute:** laptop CPU only
**Division of labor:** `[CLAUDE]` = I write & run code · `[YOU]` = you listen, analyze errors, decide fixes, own RUNLOG/NOTES/SUMMARY + the 5-min viva.

---

## 0. TL;DR strategy

Build a **small classifier over causal prosodic features** that, for every pause, outputs `p_eot`
(probability the turn is over). We do **not** tune the agent's timer — `score.py` sweeps
`(threshold × delay)` itself. **Our only job is separability**: rank true turn-ends above
mid-turn pauses, so the scorer can pick a *small* delay while interrupting ≤ 5% of turns.

1. Reproduce baseline → know the number to beat.
2. Ship an MVP fast (v1 features + gradient-boosting + a real `predict.py`) → be submittable.
3. Iterate features, measured by **grouped-CV AUC** and the **real scorer delay**.
4. **Listen to the worst errors** (your job) → every fix goes in `RUNLOG.md`.
5. Freeze, regenerate predictions for both languages, push **before the deadline**, submit the URL.

---

## 1. What "good" means — the metric (read carefully)

Scorer (`score.py`) simulates a live agent. For each candidate operating point `(threshold t, delay d)`:

- **Hold pause** (user will resume): the agent *fires* if `p_eot ≥ t`. It's a **FALSE CUTOFF**
  only if it fires **and** `d < pause_duration` (agent would speak before the user resumes).
  Any turn with ≥1 false cutoff is an **interrupted turn**.
- **True EOT pause**: response delay = `d` if we fire, else the **1.6 s timeout**.

It reports the **minimum mean response delay achievable while interrupting ≤ 5% of turns.**
Lower is better. It also prints a diagnostic **AUC** (our fast proxy).

**Consequences that drive every design choice:**

- **We never choose the threshold/delay** — the scorer does. So *calibration of `p_eot` doesn't
  matter; ranking does.* Optimize **AUC / separation**.
- **Enemy = long HOLD pauses we rank high.** A high-`p_eot` hold with a long duration becomes a
  false cutoff. We can't see duration (it's the future), so **prosody must push hesitation /
  mid-thought pauses DOWN**. These are often the long ones (thinking, listing, filler).
- **True EOTs must be ranked high enough** that a small `d` still catches them without >5%
  cutoffs — otherwise they hit the 1.6 s timeout and wreck the mean.
- Net: maximize AUC, and specifically **separate final-prosody (falling pitch, energy decay,
  final lengthening) from continuation-prosody (level/rising pitch, sustained energy).**

### Established reference points (already measured on the provided data)

| System | English delay | Hindi delay | AUC |
|---|---|---|---|
| Silence-only baseline (**must beat**) | **1600 ms** | **850 ms** | ~0.50 |
| Starter `train.py` (3 toy features) | — | — | 0.60 / 0.63 |

**Data:** English + Hindi, **100 turns / 248 pauses each**, 16 kHz mono, turns ~15–17 s.
Labels: **148 hold / 100 eot**, exactly **one eot per turn** (its final pause).
**Final grade uses a HIDDEN test set — "mostly Hindi"** — so we train language-agnostic and
validate Hindi hard.

---

## 2. Non-negotiable rules (violation = disqualified / zeroed)

1. **Causality.** Features for a pause may use **only** `audio[0 : pause_start]`.
   - ❌ Never use audio after the pause. ❌ Never use `pause_end` or **pause duration** (future).
   - ✅ `pause_start` and `pause_index` are known at decision time — allowed.
   - Keep **one** feature module imported by both train & predict, with causality asserted in code
     and commented — graders **read the feature code**.
2. **`predict.py` is the deliverable, not `train.py`.** It must run *unmodified*:
   `python predict.py --data_dir <folder> --out predictions.csv`, on a folder it has never seen,
   and **load a saved model** (no refitting).
3. **No pretrained models / weights / external data.** Allowed libs only: numpy, scipy,
   scikit-learn, pandas, librosa, PyTorch. (No Whisper/wav2vec/Silero/webrtcvad/HF/APIs.)
4. **Split by turn** in all validation (`GroupShuffleSplit`/`GroupKFold` on `turn_id`) — never leak
   a turn across train/test.
5. **CPU only.** No GPU, no cloud training.
6. **Coverage:** `predictions.csv` must contain a row for **every** `(turn_id, pause_index)` in the
   folder's `labels.csv`, `p_eot` numeric in `[0,1]` — else the scorer aborts with "missing prediction".

---

## 3. Repository layout & environment

Work inside the venv from setup: `source ~/speedrun/env/bin/activate` (prompt shows `(env)`).

```
plivo-ml-23EC39027/
├── plan.md                 # this file
├── README.md               # how to reproduce (train + predict + score)
├── eot_features.py         # SINGLE SOURCE OF TRUTH for causal features  [CLAUDE]
├── features.py             # starter audio utilities (kept, may extend)
├── train.py                # build dataset → CV → fit → save model.pkl    [CLAUDE]
├── predict.py              # load model.pkl → predict on ANY folder       [CLAUDE]
├── score.py                # official scorer (UNCHANGED)
├── model.pkl               # saved {scaler, classifier, feature_names}    [artifact]
├── predictions_english.csv # deliverable #3 (both languages)
├── predictions_hindi.csv
├── RUNLOG.md               # one entry per scoring run — GRADED            [YOU]
├── NOTES.md                # ≤10 sentences                                [YOU]
├── SUMMARY.html            # agent-generated summary                       [CLAUDE draft, YOU own]
├── tools/
│   └── dump_errors.py      # extract worst FP/FN wav clips for listening   [CLAUDE]
└── .gitignore              # ignore eot_data/ (big wavs), __pycache__, *.wav
```

**Do NOT commit the raw audio** (`eot_data/`) — it's provided input, bloats the repo, and the
grader runs `predict.py` on their own folder. Commit code + `model.pkl` + predictions + docs.

### Phase-0 setup commands `[CLAUDE]`
```bash
cd ~/plivo-ml-23EC39027
cp ~/Downloads/eot_handout/starter/{features.py,score.py} .
cp -r ~/Downloads/eot_handout/eot_data .            # local only; git-ignored
printf 'eot_data/\n__pycache__/\n*.pyc\n*.wav\n' > .gitignore
git init && git add -A && git commit -m "chore: scaffold repo, starter utils, plan"
# create the PUBLIC remote (gh already authenticated as tusharanshu-pandey):
gh repo create plivo-ml-23EC39027 --public --source=. --remote=origin --push
```

---

## 4. Solution architecture (pipeline)

```
labels.csv row ─▶ load_wav (cached per file) ─▶ audio[0:pause_start]
                                                     │
                                    eot_features.extract_features()  ← causal, single source
                                                     │  (fixed-length vector + FEATURE_NAMES)
                                                     ▼
                              StandardScaler ─▶ classifier.predict_proba ─▶ p_eot
                                                     │
                                     write turn_id,pause_index,p_eot ─▶ predictions.csv ─▶ score.py
```

- **train.py**: builds `X, y, groups` over pooled English+Hindi, does grouped-CV to report honest
  AUC + **out-of-fold** predictions (fed to `score.py` for an unbiased delay estimate), then refits
  on all data and **saves `model.pkl`**.
- **predict.py**: loads `model.pkl`, walks the folder's `labels.csv`, extracts the *same* features,
  writes predictions. **No fitting.**

---

## 5. Feature engineering `[CLAUDE codes]` · `[YOU decide what helps]`

All features from the causal window(s) before `pause_start`. Use a **local window (~1.5 s)** for
fine prosody and a **longer look-back (~3–4 s)** for trends, plus cheap context. Normalize
pitch/energy to be **speaker-relative** (stats over audio-so-far) so English & Hindi and different
speakers are comparable.

| Group | Feature | Rationale (EOT signal) |
|---|---|---|
| **Pitch (F0)** | Slope of F0 over last voiced region (linear fit) | Statements **fall**; continuations stay level / **rise** |
| | Final F0 − median F0 (speaker-relative) | Terminal drop = completion |
| | F0 range, F0 std in window | Flattening near end vs animated continuation |
| | Voiced fraction in last 0.5 s | Trailing voicing pattern |
| **Energy** | Slope of frame energy (dB) into the pause | **Decay** into silence = end |
| | Last-frame energy − window mean | Trailing loudness drop |
| | Energy std / range | Fade vs abrupt stop |
| **Timing / rate** (causal) | Duration of final continuous voiced run | **Final-syllable lengthening** = end |
| | Voiced-to-total frame ratio | Speaking density / trailing-off |
| | # voiced segments in window (syllable-rate proxy) | Slowing down before end |
| **Spectral** (librosa) | Spectral centroid & rolloff trend | Voice darkens/relaxes at end |
| | Zero-crossing rate (last frames) | Voicing/fricative cues |
| | (optional) MFCC means over last window | Timbre; adds dims — add only if it helps CV |
| **Context** | `pause_index` | Deeper pauses more likely EOT |
| | `pause_start` (time into turn) | Turn-so-far elapsed (legal: ≤ pause_start) |

**Rules for this section:**
- ⚠️ **No language feature** — the hidden folder gives no language label; keep features
  language-agnostic. (We *may* probe a language flag in CV for analysis, but the shipped model
  must not need it.)
- Guard short/empty segments → return a safe zero/low-`p` vector (already patterned in starter).
- Every feature is a pure function of `audio[0:pause_start]`, `pause_start`, `pause_index`.

---

## 6. Model plan `[CLAUDE]`

- **Preprocess:** `StandardScaler` (fit on train only), persisted in `model.pkl`.
- **Core model:** `HistGradientBoostingClassifier` (or `GradientBoosting`) — handles nonlinear
  prosody interactions, robust on ~500 pooled samples. Baseline sanity: `LogisticRegression`
  (`class_weight="balanced"`) for a linear reference in RUNLOG.
- **Class balance:** 100 eot vs 148 hold — mild; use `class_weight`/`sample_weight` or leave and
  rely on ranking (AUC is imbalance-robust).
- **Training set:** **pool English + Hindi** (more data; hidden test mostly Hindi ⇒ cross-lingual
  prosody generalization). Report per-language dev numbers separately.
- **Stretch (only after MVP + solid features):**
  - Tiny **PyTorch MLP** (2 layers) or `MLPClassifier`, early-stopped on grouped val.
  - Probability **calibration** (isotonic) — nice for interpretability, *won't* change the scorer
    result (ranking-invariant), so deprioritize.
  - **Ambition that scores points (they explicitly reward it):** a frame-level **sequence model**
    (e.g., small **TCN/GRU** over per-frame prosody up to `pause_start`) instead of hand-pooled
    stats. If it loses, **explain why in RUNLOG** — a documented failed-then-understood experiment
    beats safe tuning.

---

## 7. Validation methodology `[CLAUDE runs]` · `[YOU read]`

1. **Grouped CV** (`GroupKFold(5)` on `turn_id`) → mean **AUC** (fast proxy, target ≫ 0.63).
2. **Out-of-fold (OOF) predictions** → run through the real scorer for an **unbiased delay**
   (train never sees the turn it predicts). This is the number we trust.
3. **Per-language report**: train pooled, but print English delay, Hindi delay, and
   **train-on-English→test-on-Hindi** (and vice-versa) to gauge the cross-lingual gap that the
   hidden (mostly-Hindi) set will punish.
4. **Every run's numbers go straight into `RUNLOG.md`** with the one thing that changed.

Acceptance gates:
- **MVP gate:** beat both baselines (< 1600 ms EN, < 850 ms HI) with a real saved model.
- **Good gate:** AUC ≥ ~0.85 and Hindi delay comfortably below 850 ms, cross-lingual gap small.

---

## 8. Error analysis loop — **YOUR highest-value work** `[YOU]`

This is where the grade lives (beyond the coding-agent baseline). I'll give you the tool; you bring
the ears and the judgment.

- I run `tools/dump_errors.py` → it writes the worst **false cutoffs** (holds we scored high) and
  worst **misses** (eots we scored low) as short wav clips + a table (turn, pause, p_eot, our error).
- **You listen** to ~10–15 of each and answer, per clip:
  - What did the speaker's pitch/energy actually do? (falling? rising? level?)
  - Was it a hesitation, a list ("aur ek…", "and then"), a backchannel, code-switch, breath?
  - What feature *should* have caught it and didn't?
- You dictate the fix (new feature / window change / re-weighting); I implement; we re-score;
  you log the before/after in RUNLOG. Repeat until diminishing returns.

Likely failure modes to expect (Hindi-heavy): non-final **rising** continuations, **filler-lengthened**
holds, list intonation, and **code-switch** boundaries.

---

## 9. `predict.py` contract `[CLAUDE]`

```
python predict.py --data_dir <folder> --out predictions.csv
```
- Reads `<folder>/labels.csv` (same schema); for each row loads the wav (cache per file), calls the
  **shared** `eot_features.extract_features`, applies the **loaded** scaler+model, writes
  `turn_id,pause_index,p_eot`.
- **Must not** import training data, refit, or use `label`/`pause_end`.
- Robust to any turn count; safe fallback for short/edge segments; deterministic.
- Sanity check before submitting: run it on both provided folders and confirm `score.py` accepts the
  output with full coverage.

---

## 10. Deliverables checklist (all five required)

- [ ] **`predict.py`** — loads saved model, correct CLI, runs on unseen folder. `[CLAUDE]`
- [ ] **`predictions.csv` for BOTH languages** (`predictions_english.csv`, `predictions_hindi.csv`).
      *Note the submission form/schema wording; provide both.* `[CLAUDE]`
- [ ] **`RUNLOG.md`** — one entry per scoring run: score, what changed, why (GRADED). `[YOU]`
- [ ] **`NOTES.md`** — ≤ 10 sentences: what signal the model uses, where it fails, one-more-day plan. `[YOU]`
- [ ] **`SUMMARY.html`** — solution, results, graphs, summary of the MD files, human-vs-agent split,
      why it beats the status quo. `[CLAUDE drafts, YOU own the claims]`
- [ ] **Modified code** (`eot_features.py`, `train.py`, working `predict.py`) + `model.pkl`.

### RUNLOG.md entry template
```
### Run N — <one-line hypothesis>
- Changed: <the single thing>
- Dev (grouped OOF): AUC=___  | EN delay=___ ms | HI delay=___ ms
- Heard (from error clips): <what you noticed>
- Conclusion / next: <keep or revert, why>
```

### NOTES.md skeleton (≤10 sentences)
Signal used · best config & numbers · where it fails · what one more day buys · causality guarantee.

---

## 11. Git & submission workflow `[CLAUDE runs, YOU confirm]`

- **Commit cadence:** after each milestone — scaffold, MVP, each feature/model change that moved the
  score. Small, message-per-change (feeds RUNLOG too). `git add -A && git commit -m "..." && git push`.
- **Deadline rule is absolute:** *the last commit must be pushed BEFORE the deadline; any commit
  after disqualifies.* → Do the **final push with a safety buffer**, then STOP committing.
- **Verify** at `https://github.com/tusharanshu-pandey/plivo-ml-23EC39027`: latest commit present,
  repo is **public**, `predict.py` + `model.pkl` + both predictions + docs are there.
- Paste the repo URL into the Google Form. Stay in the mandatory **Zoom** the entire time.

---

## 12. Suggested phasing (do it properly; MVP keeps us safe)

| Phase | Goal | Owner | Exit state |
|---|---|---|---|
| 0 · Setup | Repo, venv, copy starter, reproduce baseline | `[CLAUDE]` | baseline reproduced, repo pushed |
| 1 · **MVP** | v1 features + GBM + real `predict.py`, score both langs, commit | `[CLAUDE]` | **beats baseline, submittable** |
| 2 · Features | Add pitch/energy/timing/spectral/context + speaker-norm; grouped-CV | `[CLAUDE]` + `[YOU]` | AUC climbing, logged |
| 3 · **Listen** | Dump worst FP/FN, you analyze, we fix | `[YOU]` lead | error-driven fixes logged |
| 4 · Polish | Stretch model / cross-lingual robustness; pick best by OOF score | `[CLAUDE]` | best model frozen |
| 5 · Docs | RUNLOG finalized, NOTES, SUMMARY.html + graphs | `[YOU]` + `[CLAUDE]` | all 5 deliverables done |
| 6 · Ship | Regenerate both predictions, commit, **push before deadline**, verify, submit URL | `[CLAUDE]` + `[YOU]` | URL in form, Zoom on |

**MVP checkpoint (end of Phase 1): you are always in a submittable state.** Everything after only
raises the score.

---

## 13. Risk register / gotchas

- **Causality leak** (using `pause_end`/duration/future audio) → single-source `eot_features.py`,
  causality asserted + commented; graders read it.
- **Refit in `predict.py`** → forbidden; always load `model.pkl`.
- **Turn leakage across split** → group by `turn_id` everywhere.
- **Overfitting the operating point** → don't hardcode threshold/delay; the scorer sweeps. Optimize
  ranking (AUC/OOF delay).
- **Language over-reliance** → no language feature; validate Hindi and cross-lingual explicitly.
- **Long-hold false cutoffs** → ensure hesitation/list/filler holds get low `p_eot` via non-final
  prosody features.
- **Coverage/format** → every `(turn_id, pause_index)` present, `p_eot ∈ [0,1]`.
- **Repo bloat / accidental data commit** → `.gitignore eot_data/`, `*.wav`.
- **Deadline miss** → final push with buffer; verify commit timestamp < deadline; then stop.
- **Reproducibility** → fixed `random_state`; `model.pkl` bundles scaler+classifier+feature_names.

---

## 14. 5-minute discussion (viva) prep `[YOU]`

Be ready to explain, in your words:
- **The signal:** why falling F0 + energy decay + final lengthening ⇒ EOT; what continuations look like.
- **Causality:** exactly how you guaranteed no future leak (window = `audio[0:pause_start]`, no duration).
- **The metric:** the 5%-cutoff ↔ delay tradeoff; why long high-`p` holds are the danger; why AUC is the proxy.
- **Cross-lingual:** why you pooled EN+HI, how Hindi prosody differed, what the hidden-set gap looked like.
- **An ambitious experiment** you tried, whether it won or lost, and **why** (this is explicitly rewarded).
- **Human vs agent:** which insights/fixes came from *your* listening (name specific clips), vs what I scaffolded.
- **One more day:** streaming per-frame EOT head, richer voicing, speaker adaptation, more data.

---

## 15. Division of labor (recap)

- `[CLAUDE]`: repo setup, `eot_features.py`, `train.py`, `predict.py`, `tools/dump_errors.py`,
  all runs/scoring, SUMMARY.html draft + graphs, git/push.
- `[YOU]`: listen to error clips, decide feature/model fixes, write **RUNLOG.md** and **NOTES.md**,
  own SUMMARY's claims, and carry the **viva**. This is where your grade beats the coding-agent floor.

---

### Immediate next action
Say **"go"** and I'll execute **Phase 0 + Phase 1**: scaffold the repo, write `eot_features.py`
(v1), `train.py`, and `predict.py`, train the model, and score both languages — landing you an
MVP that beats the baseline and is pushed to GitHub. Then we enter the listen-and-fix loop.
