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

### Run 4 — v3 features from Hindi error-listening
- **Heard (human, native Hindi, 20 worst Hindi clips):** true turn-ends **fade out** (*"shukriya", "aa jaungi", "karlunga", "nau"*); the dangerous long holds **stop abruptly** mid-dictation (*"panch… likh lijiye", "note kariye number…"*) — speaker fades when done, cuts abruptly when continuing a number/list. One "hold" (`hi__060`) was a **click with no speech**, yet scored 0.88. Several holds sounded *finished* even to a native speaker → **label noise** caps AUC.
- **Changed:** +4 features — `e_fade_slope` (energy slope over last 500 ms), `e_last_minus_winmin` (faded to the window floor?), `speech_presence` (silence/click guard), `local_max_minus_spkmax`.
- **Dev:** OOF AUC 0.685; EN **1209 ms** (↓ from 1232), HI **850 ms** (unchanged).
- **Conclusion:** fade/silence cues help English slightly; Hindi unmoved — the blocking holds are mid-dictation pauses that need the *words* (verb-final Hindi), which acoustics can't recover.

### Run 5 — is Hindi better with a Hindi-focused model? (hidden set is mostly Hindi)
- **Hypothesis:** since grading is mostly Hindi, a Hindi-only / Hindi-upweighted model might beat the pooled one on Hindi.
- **Changed:** trained Hindi-only vs pooled; compared Hindi OOF.
- **Dev:** pooled → Hindi AUC **0.706**; Hindi-only → **0.635** (worse). Delay 850 either way.
- **Conclusion:** **pooling English+Hindi *helps* Hindi** (shared prosody transfer) — design validated. Hindi's 850 ms is a genuine ceiling for acoustic-only EOT here, not a tuning miss.

### Run 6 — guard: never fire on a pause with no audible evidence
- **Found (agent audit):** pauses with < 0.2 s of history get the all-zero feature vector, which the trained model maps to **p_eot = 0.631** — above every operating threshold, i.e. a guaranteed fire with zero evidence. **0/496 dev pauses** hit this path (so dev scores are unchanged), but one early-turn pause per hidden file could burn the 5% cutoff budget.
- **Changed:** `predict.py` emits `LOW_CONTEXT_P = 0.02` for such pauses (single-sourced in `eot_features.low_context`). Rationale: a false fire costs a whole interrupted turn; waiting costs at most the 1.6 s timeout on a rare very-early eot.
- **Dev:** EN/HI unchanged (no dev row affected) — pure hidden-set downside protection.

### Run 7 — measure evaluation noise; repeated grouped CV becomes the harness
- **Hypothesis (agent audit):** with 100 turns/language, a single 5-fold grouped split has delay noise comparable to our run-to-run deltas.
- **Changed:** `train.py` now averages over **5 random grouped fold assignments** (`--cv_seeds`) and reports mean ± std; decisions must clear the std. Also prints a Hindi-weighted (3:1) selection metric since the hidden set is mostly Hindi.
- **Dev:** EN **1230 ± 32 ms**, HI **834 ± 22 ms**, AUC 0.654 ± 0.011. Two candidate "wins" (duration-weighted holds: single-split EN 1128; causal pause-history features: single-split EN 1158/AUC 0.699) **evaporated** under repeated CV — both were fold noise, neither is shipped.
- **Conclusion:** the honest headline is 1230 ± 32 / 834 ± 22, not the single-split 1209/850; earlier ±20 ms deltas (e.g. v3's EN 1232→1209) are inside noise. Single-split numbers are no longer trusted.

### Run 8 — pyin pitch tracker: better F0 ≠ better metric (not shipped)
- **Hypothesis:** the top-weighted feature (`f0_final_pctl`) sits on a bare autocorrelation tracker; a cleaner algorithmic tracker (`librosa.pyin` — YIN-based, no pretrained weights) should sharpen every pitch feature.
- **Changed (experiment only):** pyin for the 1.5 s local window (same 10 ms hop), autocorr kept for the speaker baseline. The cleanup was real: voiced runs defragmented (7.5 → 3.3 segments/window), voiced fraction 0.73 → 0.63.
- **Dev (5 fold seeds):** AUC 0.654 → **0.667** (+~1σ), but delay EN **1256 ± 31** / HI **840 ± 16** vs base 1230 ± 32 / 834 ± 22 — no win where it counts, and ~0.3 s/pause slower inference.
- **Conclusion:** **not shipped.** Cleaner pitch raised *average* ranking quality but didn't move the specific pauses that set the operating point — more evidence the metric lives in top-of-ranking purity (EN) and long-hold separation (HI), not AUC.

### Verdict (numbers = mean ± std over 5 grouped fold assignments; see Run 7)
- **English 1230 ± 32 ms** vs 1600 baseline — decisive ~23% win, robust across fold seeds.
- **Hindi 834 ± 22 ms** ≈ 850 baseline — the scorer's Hindi operating point degenerates to the silence-timer policy (threshold 0.05 fires on every pause); the current features can't yet separate true ends from the *long* holds that set the delay. Label noise + lexical/verb-final dependence cap the acoustics.
- Overall OOF AUC **0.654 ± 0.011**. Model: LR + bagged-GBM soft-voting ensemble on 30 causal features, pooled training, plus a low-context safety guard in `predict.py` (Run 6).
- Historical single-split numbers earlier in this log (e.g. EN 1209/1232) are kept as recorded but should be read with Run 7's ±30–45 ms noise bar in mind.

---

## Human vs coding-agent split (for honesty + SUMMARY)
- **Coding agent (Claude):** all code (`eot_features.py`, `train.py`, `predict.py`, error-dump tool), the runs, the model bake-off, coefficient analysis.
- **Human (me):** listened to the worst error clips and identified the number/address flat-ending failure mode and the fade-vs-cut distinction — which is what turned v1 (AUC 0.659, generic prosody) into v2's declination/rhythm features (the top-weighted signals). Decided to prioritize the English misses and to trust the robust HI=850 over the lucky 808.
- **Second pass (agent-driven audit, human-approved):** scorer headroom analysis, the zero-vector robustness fix (Run 6), the repeated-CV noise measurement that corrected the headline numbers (Run 7), and two experiments rejected as fold noise (duration-weighted holds; causal pause-history features).

## Next levers (if continued)
- Crack Hindi where the metric actually lives: separate true ends from **long (>0.5 s) holds only** — 83% of Hindi holds are too short to ever cause a false cutoff at a 0.5 s delay, yet the model currently spends capacity on them. Targeted listening on those long mid-dictation holds; needs human ears.
- English: the mean is dominated by the ~54% of eots that miss the operating threshold and pay the 1.6 s timeout — dump and listen to *marginal* errors near the threshold, not the global worst.
- Frame-level sequence model (small GRU/TCN over per-frame prosody up to `pause_start`) instead of pooled stats — ambitious; would need a careful causal implementation.
