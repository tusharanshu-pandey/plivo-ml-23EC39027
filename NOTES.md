# NOTES

1. The model scores each pause with `p_eot` from 30 **causal** prosodic features over the audio strictly before the pause (never `pause_end`/duration).
2. Best config: a soft-voting ensemble of a regularized LogisticRegression + a bagged shallow GradientBoosting, trained on **pooled** English+Hindi (pooling *helps* Hindi — Hindi-only training scored AUC 0.635 vs 0.706 pooled).
3. Honest grouped out-of-fold result: **English 1209 ms** (baseline 1600) and **Hindi 850 ms** (baseline 850), overall AUC 0.685.
4. Dominant signals: **pitch declination-to-floor** (`f0_final_pctl`), **voiced-rhythm regularity** (`voiced_seg_dur_std`), **energy fade-to-floor**, and how deep the pause is in the turn.
5. These came from listening to errors: turn-ends on **numbers / addresses / codes** end *flat* (no pitch fall), so declination-to-floor — not slope — marks them done; speakers **fade** when finished and **cut abruptly** when continuing a dictation.
6. Where it fails: **Hindi is at its acoustic ceiling** — verb-final Hindi ends on a short auxiliary (*hai/hoon/lunga*) whose identity is lexical, not acoustic, and the ≤5% false-cutoff budget blocks a smaller delay despite AUC ≈ 0.71 on Hindi.
7. It also fails on genuinely ambiguous endings and on **label noise** (holds that even a native Hindi speaker hears as finished).
8. "Abruptness" is deliberately *not* trusted alone — it appears in both true ends and mid-turn holds; only fade-to-floor discriminates.
9. Human vs agent: the agent wrote all code and ran the search; the human's English + native-Hindi listening supplied the flat-ending, fade-vs-cut, and silence-artifact insights that produced the top-weighted features.
10. With one more day: a causal frame-level sequence model (GRU/TCN) over per-frame prosody, and a lightweight from-scratch final-syllable/keyword acoustic detector to recover the verb-final Hindi cue and finally beat 850 ms.
