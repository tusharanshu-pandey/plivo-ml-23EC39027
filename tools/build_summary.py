"""Generate SUMMARY.html (self-contained, inline SVG charts) from the real
model + data. Run: python tools/build_summary.py
"""
import os, sys, csv, html
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from train import build_dataset, make_model, random_group_folds, DEFAULT_DIRS
from eot_features import FEATURE_NAMES
import score as scorer

X, y, groups, langs, keys, dir_of = build_dataset(DEFAULT_DIRS)
langs = np.array(langs)

# OOF probabilities over repeated fold seeds (single-split delay noise is
# ±30-45 ms on 100 turns/lang — see RUNLOG Run 7; we report mean ± std)
SEEDS = 5
oofs = []
for s in range(SEEDS):
    o = np.zeros(len(y))
    for tr, te in random_group_folds(groups, s):
        m = clone(make_model()); m.fit(X[tr], y[tr]); o[te] = m.predict_proba(X[te])[:, 1]
    oofs.append(o)
oof = oofs[0]                                   # seed-0 OOF for the distribution plot
auc_seeds = [roc_auc_score(y, o) for o in oofs]
auc, auc_sd = float(np.mean(auc_seeds)), float(np.std(auc_seeds))

def delay_for(pred, d):
    idx = [i for i in range(len(dir_of)) if dir_of[i] == d]
    with open("scratch/_sum.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["turn_id", "pause_index", "p_eot"])
        for i in idx: w.writerow([keys[i][0], keys[i][1], f"{pred[i]:.6f}"])
    return scorer.score(os.path.join(d, "labels.csv"), "scratch/_sum.csv")

ones = np.ones(len(y))
res = {}
for d in DEFAULT_DIRS:
    lang = os.path.basename(d.rstrip("/"))
    vals = [delay_for(o, d)["latency"] * 1000 for o in oofs]
    res[lang] = {"model": float(np.mean(vals)), "model_sd": float(np.std(vals)),
                 "base": delay_for(ones, d)["latency"] * 1000}

# LR standardized coefficients (interpretability)
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")).fit(X, y)
coef = lr.named_steps["logisticregression"].coef_[0]
order = np.argsort(-np.abs(coef))[:14]
HUMAN = {"f0_final_pctl","f0_final_minus_spkmin","e_final_minus_spkmax","e_tail_drop",
         "n_voiced_segments","voiced_seg_dur_mean","voiced_seg_dur_std","last_voiced_offset",
         "e_fade_slope","e_last_minus_winmin","speech_presence","local_max_minus_spkmax"}

# ---------------- SVG helpers ----------------
def svg_delays():
    W, H, pad = 460, 220, 40
    groups_ = ["english", "hindi"]; maxv = 1700
    bw = 46; gap = 30; x0 = 70; svg = [f'<svg viewBox="0 0 {W} {H}">']
    for gi, g in enumerate(groups_):
        gx = x0 + gi * 190
        for bi, (key, col, lab) in enumerate([("base", "#c98b3a", "baseline"), ("model", "#2a9d8f", "model")]):
            v = res[g][key]; h = (v / maxv) * (H - 2 * pad)
            x = gx + bi * (bw + 12); yb = H - pad - h
            svg.append(f'<rect x="{x}" y="{yb:.0f}" width="{bw}" height="{h:.0f}" fill="{col}" rx="3"/>')
            svg.append(f'<text x="{x+bw/2:.0f}" y="{yb-6:.0f}" text-anchor="middle" font-size="12" fill="#333">{v:.0f}</text>')
        svg.append(f'<text x="{gx+bw:.0f}" y="{H-pad+16:.0f}" text-anchor="middle" font-size="13" fill="#333">{g}</text>')
    svg.append(f'<line x1="{x0-14}" y1="{H-pad}" x2="{W-10}" y2="{H-pad}" stroke="#ccc"/>')
    svg.append('<text x="14" y="20" font-size="12" fill="#666">delay (ms) — lower is better</text>')
    svg.append('<rect x="300" y="8" width="12" height="12" fill="#c98b3a"/><text x="316" y="18" font-size="11">baseline</text>')
    svg.append('<rect x="300" y="24" width="12" height="12" fill="#2a9d8f"/><text x="316" y="34" font-size="11">model</text>')
    return "".join(svg) + "</svg>"

def svg_coef():
    W, rowh, pad = 560, 22, 10; H = pad * 2 + rowh * len(order)
    mx = max(abs(coef[i]) for i in order); midx = 300; scale = 210 / mx
    svg = [f'<svg viewBox="0 0 {W} {H}">', f'<line x1="{midx}" y1="0" x2="{midx}" y2="{H}" stroke="#ddd"/>']
    for r, i in enumerate(order):
        yb = pad + r * rowh; w = abs(coef[i]) * scale
        col = "#2a9d8f" if coef[i] > 0 else "#c98b3a"
        x = midx if coef[i] > 0 else midx - w
        svg.append(f'<rect x="{x:.0f}" y="{yb:.0f}" width="{w:.0f}" height="{rowh-6}" fill="{col}" rx="2"/>')
        star = " ★" if FEATURE_NAMES[i] in HUMAN else ""
        svg.append(f'<text x="8" y="{yb+rowh-9:.0f}" font-size="12" fill="#333">{html.escape(FEATURE_NAMES[i])}{star}</text>')
        svg.append(f'<text x="{(x+w+4) if coef[i]>0 else (x-4):.0f}" y="{yb+rowh-9:.0f}" font-size="11" fill="#666" text-anchor="{"start" if coef[i]>0 else "end"}">{coef[i]:+.2f}</text>')
    return "".join(svg) + "</svg>"

def svg_sep():
    W, H, pad = 460, 200, 34; bins = np.linspace(0, 1, 11)
    he, _ = np.histogram(oof[y == 1], bins=bins); hh, _ = np.histogram(oof[y == 0], bins=bins)
    mx = max(he.max(), hh.max()); bw = (W - 2 * pad) / 10
    svg = [f'<svg viewBox="0 0 {W} {H}">']
    for b in range(10):
        x = pad + b * bw
        for arr, col, off in [(hh, "#c98b3a", 0), (he, "#2a9d8f", bw/2)]:
            h = (arr[b] / mx) * (H - 2 * pad)
            svg.append(f'<rect x="{x+off:.1f}" y="{H-pad-h:.0f}" width="{bw/2-1:.1f}" height="{h:.0f}" fill="{col}"/>')
    svg.append(f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" stroke="#ccc"/>')
    svg.append(f'<text x="{pad}" y="{H-pad+16}" font-size="11" fill="#666">p_eot 0</text>')
    svg.append(f'<text x="{W-pad-24}" y="{H-pad+16}" font-size="11" fill="#666">1</text>')
    svg.append('<text x="10" y="16" font-size="12" fill="#666">OOF p_eot distribution</text>')
    svg.append('<rect x="300" y="6" width="12" height="12" fill="#2a9d8f"/><text x="316" y="16" font-size="11">eot</text>')
    svg.append('<rect x="350" y="6" width="12" height="12" fill="#c98b3a"/><text x="366" y="16" font-size="11">hold</text>')
    return "".join(svg) + "</svg>"

HTMLDOC = f"""<!doctype html><html><head><meta charset="utf-8">
<title>EOT Detection — Summary (23EC39027)</title>
<style>
body{{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:860px;margin:32px auto;padding:0 18px;color:#1c1c1c}}
h1{{font-size:24px;margin-bottom:2px}} h2{{margin-top:30px;border-bottom:2px solid #2a9d8f;padding-bottom:4px}}
.sub{{color:#666}} table{{border-collapse:collapse;margin:10px 0}} td,th{{border:1px solid #ddd;padding:6px 12px;text-align:left}}
th{{background:#f4f4f4}} .win{{color:#2a9d8f;font-weight:600}} .card{{background:#fafafa;border:1px solid #eee;border-radius:8px;padding:14px 18px;margin:10px 0}}
code{{background:#f0f0f0;padding:1px 5px;border-radius:3px}} .star{{color:#2a9d8f}}
</style></head><body>
<h1>End-of-Turn Detection</h1>
<div class="sub">Tusharanshu Pandey &middot; 23EC39027 &middot; STT track &middot; CPU-only, no pretrained models</div>

<h2>Result</h2>
<p>For every pause in a user turn we output <code>p_eot</code> from <b>causal</b> prosodic features
(audio strictly before the pause). Honest grouped-by-turn out-of-fold, averaged over {SEEDS} random
fold assignments (single-split numbers proved untrustworthy — see "Evaluation noise" below), overall
AUC <b>{auc:.3f} &plusmn; {auc_sd:.3f}</b>:</p>
<table><tr><th></th><th>English</th><th>Hindi</th></tr>
<tr><td>Silence-only baseline</td><td>{res['english']['base']:.0f} ms</td><td>{res['hindi']['base']:.0f} ms</td></tr>
<tr><td>This model</td><td class="win">{res['english']['model']:.0f} &plusmn; {res['english']['model_sd']:.0f} ms</td><td>{res['hindi']['model']:.0f} &plusmn; {res['hindi']['model_sd']:.0f} ms</td></tr></table>
<div class="card">{svg_delays()}</div>
<p>Metric = mean response delay at &le;5% interrupted turns (lower is better). <b>English improves
~23%</b>; Hindi is statistically level with the (already strong) silence baseline — at the ceiling of
the <i>current features</i>, not of the data (see below).</p>

<h2>Approach</h2>
<p>A soft-voting ensemble — a regularized <b>LogisticRegression</b> + a bagged shallow
<b>GradientBoosting</b> (3 seeds) — over 30 causal features, trained on <b>pooled</b> English+Hindi.
Pooling was validated: a Hindi-only model scored worse on Hindi (AUC 0.635 vs 0.706), i.e. English
prosody transfers. We picked this over a single HistGradientBoosting that memorized the training set
(in-sample AUC 1.0) yet lost out-of-fold.</p>
<p><b>The scorer sweeps threshold &amp; delay itself</b>, so we optimize <i>separability</i>, not a
timer. That means down-ranking mid-turn holds and up-ranking true ends:</p>
<div class="card">{svg_sep()}</div>

<h2>What the features are (and where they came from)</h2>
<p>Standardized LogisticRegression weights. <span class="star">&starf;</span> = feature added because of
a <b>human error-listening</b> pass (see below), not generic prosody:</p>
<div class="card">{svg_coef()}</div>
<p>The two strongest signals are both human-driven: <code>f0_final_pctl</code> (final pitch at the
<i>bottom of the speaker's range</i> &rArr; finished, even on a flat number) and
<code>voiced_seg_dur_std</code> (steady digit-reading rhythm), plus energy <b>fade-to-floor</b>.</p>

<h2>How we got here (RUNLOG in brief)</h2>
<ol>
<li><b>v1</b> generic turn-final prosody + HistGB &rarr; AUC 0.66, but overfit (in-sample AUC 1.0).</li>
<li><b>Listened to the 12 worst errors</b> &rarr; the misses were English turn-ends on <b>numbers/addresses</b>
that end <i>flat</i> (no pitch fall); "abruptness" appeared in both classes. Added pitch
<b>declination-to-floor</b>, energy fade, list/number rhythm &rarr; linear AUC 0.62&rarr;0.66.</li>
<li><b>Model bake-off</b> &rarr; dropped the overfit booster for the LR + bagged-GBM ensemble (AUC 0.686).</li>
<li><b>Listened to the worst Hindi errors (native speaker)</b> &rarr; ends <b>fade out</b>
(<i>"shukriya", "karlunga"</i>), dangerous holds <b>cut abruptly</b> mid-dictation
(<i>"panch&hellip; likh lijiye"</i>); one "hold" was a silent click scored 0.88. Added fade-vs-cut +
a silence guard (single-split English 1232&rarr;1209 ms — later shown to be within evaluation noise).</li>
<li><b>Hindi limit diagnosed</b>: verb-final Hindi ends on a short auxiliary whose identity is
<i>lexical</i>, plus genuine label noise (holds a native speaker hears as finished). With the current
features the &le;5% budget blocks anything below the baseline delay.</li>
<li><b>Robustness guard</b>: pauses with &lt; 0.2 s of history got a zero feature vector that the
model scored <b>0.63</b> (fires at any threshold). 0/496 dev pauses hit this, but the hidden set
might — <code>predict.py</code> now emits a safe 0.02 for them.</li>
<li><b>Evaluation noise measured</b>: repeating the grouped CV over 5 random fold assignments showed
&plusmn;30&ndash;45 ms delay noise — larger than most logged deltas. Two plausible "wins"
(duration-weighted hold samples; causal pause-history features) <b>failed replication</b> and were
rejected. All headline numbers are now mean &plusmn; std across seeds.</li>
</ol>

<h2>Evaluation noise &amp; where the metric actually lives</h2>
<div class="card">
<b>Single-split numbers lie at this sample size.</b> The previously reported EN 1209 / HI 850 came from
one fortunate fold assignment; across 5 assignments the honest estimate is
EN {res['english']['model']:.0f} &plusmn; {res['english']['model_sd']:.0f} ms,
HI {res['hindi']['model']:.0f} &plusmn; {res['hindi']['model_sd']:.0f} ms.<br><br>
<b>Hindi's operating point is the silence baseline's policy</b>: the scorer picks threshold 0.05
(every pause fires) with a 0.85 s delay set purely by the handful of holds longer than 0.85 s —
despite Hindi AUC &asymp; 0.71, the ranking isn't yet usable there. The path to beating it is
separating true ends from only the <i>long</i> (&gt;0.5 s) holds: 83% of Hindi holds are shorter than
0.5 s and can never cause a false cutoff at that delay. On English the mean is dominated by the 54%
of true ends that miss the threshold and pay the full 1.6 s timeout.</div>

<h2>Human vs coding agent (honest split)</h2>
<div class="card">
<b>Coding agent (Claude Code):</b> all code (features, training, <code>predict.py</code>, the
error-dump &amp; summary tools), every run, the model bake-off, and the coefficient analysis.<br><br>
<b>Human (candidate):</b> listened to the worst error clips in <b>both English and native Hindi</b> and
identified (a) the flat number/address endings, (b) fade-out vs abrupt-cut, and (c) the silent-click
data artifact — the insights that produced the top-weighted features and turned generic-prosody v1
(AUC 0.66) into the final model. Also decided to trust robust numbers over lucky ones (Hindi 850 vs a
single-seed 808), and to keep pooled training after validating it helps Hindi.<br><br>
<b>Second pass (agent-driven audit, human-approved):</b> the headroom analysis of the scorer, the
zero-vector robustness bug and its fix, the repeated-CV noise measurement that corrected the headline
numbers, and the two rejected-as-noise experiments (duration-weighted holds, pause-history features).
</div>

<h2>Why it beats the status quo</h2>
<p>The silence-only endpointer waits a fixed timer (1600 ms English). By scoring end-of-turn from
prosody it reaches the true end ~23% faster on English at the same 5% interruption budget, and it does
so <b>causally</b> (features only from audio before the pause) so it is deployable in a live agent. The
honest limits — Hindi's lexical/verb-final dependence, dataset label noise, and the measured
&plusmn;30 ms evaluation noise — are documented rather than hidden, with a concrete next step (attack
the long-hold-vs-eot separation the metric actually depends on; a causal frame-level sequence model)
in <code>NOTES.md</code>.</p>
</body></html>"""

with open("SUMMARY.html", "w") as f:
    f.write(HTMLDOC)
print(f"wrote SUMMARY.html  (AUC={auc:.3f}, EN {res['english']['model']:.0f}ms, HI {res['hindi']['model']:.0f}ms)")
