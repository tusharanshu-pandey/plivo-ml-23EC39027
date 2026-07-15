# RUNLOG — EOT Detection

Metric = mean response delay (ms) at ≤5% interrupted turns (lower better), via `score.py`.
All model numbers are **grouped (by-turn) out-of-fold** — the honest estimate. In-sample is
meaningless here (a booster memorizes 496 samples). Baselines to beat: **EN 1600 ms, HI 850 ms**.

---

### Run 0 — silence-only baseline (reference)
- **Changed:** none (given). p_eot = 1 everywhere → agent is a pure silence timer.
- **Dev:** EN **1600 ms**, HI **850 ms**, AUC ≈ 0.50.
- **Conclusion:** number to beat. Note HI is already low — short Hindi holds make silence-timing hard to beat.

### Run 1 — v1 prosodic features + HistGradientBoosting
- **Hypothesis:** classic turn-final prosody (falling F0, energy decay, final lengthening) separates eot from hold.
- **Changed:** 18 causal features (F0 slope/terminal-drop, energy dynamics, voiced fraction, spectral shape, `pause_index`, `t_into_turn`) + HistGB classifier.
- **Dev:** OOF AUC **0.659**; EN **1237 ms** ✅, HI **809 ms** ✅ (both beat baseline).
- **Conclusion:** works, but AUC low. ⚠️ In-sample AUC = 1.0 → the booster is memorizing; suspect overfit.

### Run 2 — v2 features from human error-listening
- **Heard (human, 12 worst OOF clips):** the worst misses are **English turn-ends that end on numbers / addresses / codes** — these end **flat**, with no statement-final pitch fall, so a pitch-fall model misses them. "Stops abruptly" occurred in *both* missed-eots and false-cutoff holds → abruptness is **not** discriminative. One miss was a short "yes" after long silence (fixed 1.5 s window dilutes it).
- **Changed:** +8 features — **pitch declination-to-floor** (`f0_final_pctl`, `f0_final_minus_spkmin`), **energy fade-to-floor** (`e_final_minus_spkmax`, `e_tail_drop`), **list/number rhythm** (`n_voiced_segments`, `voiced_seg_dur_mean/std`), **last-voiced-region focus** (`last_voiced_offset`).
- **Dev:** linear-model OOF AUC **0.619 → 0.664**; HistGB flat (0.655) — more features made the booster overfit *more*.
- **Conclusion:** the new (human-driven) features carry real *linear* signal; the model is now the bottleneck, not the features.

### Run 3 — model bake-off → lock robust ensemble
- **Hypothesis:** on ~500 samples a regularized/ensembled model generalizes better than a deep booster.
- **Changed:** compared LR (C sweep), RF, ExtraTrees, HistGB(reg), GBM(small), ensembles, over grouped CV. Locked **soft-voting ensemble = regularized LogisticRegression + bagged shallow GBM (3 seeds)**. Dropped HistGB.
- **Dev:** OOF AUC **0.686**; EN **1232 ms** ✅ (~23% under baseline), HI **850 ms** (= baseline).
- **Evidence the human insight drove the model:** top standardized LR coefficients are `f0_final_pctl` **−0.62** (final pitch at the speaker's floor ⇒ eot, even on flat numbers) and `voiced_seg_dur_std` **−0.50** (steady digit-reading rhythm ⇒ eot) — both from Run 2's listening.
- **Conclusion:** English decisively beaten. **Hindi sits at its acoustic floor**: AUC on Hindi is ~0.72, but the ≤5% false-cutoff constraint + short Hindi holds leave no room for a smaller delay. A single seed occasionally hit 808 ms on HI — that was variance, not signal; the ensemble reports the robust 850 ms.

---

## Human vs coding-agent split (for honesty + SUMMARY)
- **Coding agent (Claude):** all code (`eot_features.py`, `train.py`, `predict.py`, error-dump tool), the runs, the model bake-off, coefficient analysis.
- **Human (me):** listened to the worst error clips and identified the number/address flat-ending failure mode and the fade-vs-cut distinction — which is what turned v1 (AUC 0.659, generic prosody) into v2's declination/rhythm features (the top-weighted signals). Decided to prioritize the English misses and to trust the robust HI=850 over the lucky 808.

## Next levers (if continued)
- Crack Hindi: targeted listening on **medium-duration Hindi holds (~0.4–0.85 s)** — those are what block a smaller delay. Needs human ears.
- Frame-level sequence model (small GRU/TCN over per-frame prosody up to `pause_start`) instead of pooled stats — ambitious; would need a careful causal implementation.
