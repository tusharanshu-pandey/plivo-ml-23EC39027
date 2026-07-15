# NOTES

1. The model scores each pause with `p_eot` from 26 **causal** prosodic features over the audio strictly before the pause (never `pause_end`/duration).
2. Best config: a soft-voting ensemble of a regularized LogisticRegression + a bagged shallow GradientBoosting, trained on pooled English+Hindi.
3. Honest grouped out-of-fold result: **English 1232 ms** (baseline 1600) and **Hindi 850 ms** (baseline 850), overall AUC 0.686.
4. The dominant signals are **pitch declination-to-floor** (`f0_final_pctl`) and **voiced-rhythm regularity** (`voiced_seg_dur_std`), plus how deep the pause is in the turn.
5. These came from listening to errors: turn-ends on **phone numbers / addresses / codes** end *flat* (no pitch fall), so declination-to-floor — not slope — is what marks them done.
6. Where it fails: **Hindi is at its acoustic floor** — short Hindi holds make silence-timing already near-optimal, and the ≤5% false-cutoff budget blocks a smaller delay despite AUC ≈ 0.72 on Hindi.
7. It also fails on genuinely ambiguous endings (a curt "yes"/number that even a human hears as unfinished) and on turns cut short by line interruptions.
8. "Abruptness" is deliberately *not* trusted — it appears equally in true ends and mid-turn holds.
9. Human vs agent: the agent wrote all code and ran the search; the human's listening supplied the flat-number-ending insight that produced the two top-weighted features.
10. With one more day: a causal frame-level sequence model (GRU/TCN) and targeted work on medium-length Hindi holds to finally beat 850 ms.
