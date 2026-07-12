# Pre-registered replication: k=160 headline (fiction corpus, fresh fit)

Declared 2026-07-11, **before running**. This commit is the declaration; results will be
reported whatever they are.

## Motivation
Reviewers flagged (a) the 160-head operating point was chosen after seeing the damage curve,
and (b) one seed / one calibration sequence per protocol, with chunk-bootstrap CIs covering
evaluation noise only (measured fit variance: refitting on different hardware moved the
unhealed cost by +0.8 nats). This replication re-runs the headline with a fresh calibration
sequence and fresh seeds, everything else fixed in advance.

## Protocol (fixed)
- Model: Qwen/Qwen3-0.6B, fp32, eager attention (local MPS).
- Corpus: `ministral_corpus.txt` (fiction), unchanged.
- **Fresh calibration sequence**: `SUB_CALIB_OFFSET=70000` (chars 70,000–81,000; disjoint from
  every heal-training chunk [20000+40000k, +8000] and every held-out eval chunk
  [40000+80000k, +8000]).
- **Fresh seeds**: `SUB_SEED_RND=137` (repeated-random selection sequence),
  `SUB_SEED_HEAL=138` (heal-time random-repeat chunks).
- Pipeline: `replace_rich.py` (refit all 448 heads on the fresh calibration sequence, solo
  costs) → `rnd_solo.py` → `mlp_prosthesis.py` → `heal_shuffle.py` (real + shuffled, 20-epoch
  fresh heals, lr 3e-4) → `heal_intact_baseline.py` (offset).
- Head selection: combined natural+random solo-cost rank, k=160 (the published operating point).
- MLP selection: 6 solo-cheapest, **with the declared guard** (the WikiText lesson): if any
  selected layer is < 6 or > 21, substitute the middle band [9,10,11,12,13,14] instead.
- Evaluation: the same 8 held-out offsets as the published runs (0 mod 40000, disjoint from
  training; never used for early stopping).

## Success criteria (declared in advance)
1. Real-code healed damage within **±0.15 nats** of the published fresh-heal value **+0.705**.
2. Shuffle separation: shuffled − real ≥ **+1.0 nats**.
3. Intact-heal offset within **±0.05** of the published **−0.242**.

Failing any criterion will be reported as a failed replication in both paper versions.

## Results (run 2026-07-11, local M4, ~55 min; log in prereg_run/prereg.log)
- Fresh-calibration fit: attention R² median 0.82 (250 heads ≥ 0.8) — matches original 0.82/248.
- MLP guard **fired**: fresh solo scan ranked L5 among the cheapest six (outside the declared
  [6, 21] band), so the middle band [9–14] was used, per the declared rule. The solo MLP ranking
  is unstable across calibration sequences; the guard caught it mechanically.
- real code + heal: **+0.682** [+0.605, +0.755] — vs published +0.705, Δ = 0.023 ≤ 0.15 ✓
- shuffled code + heal: **+4.281** [+3.499, +5.139] — separation +3.60 ≥ +1.0 ✓
- intact-heal offset: **−0.240** [−0.273, −0.204] — vs published −0.242, Δ = 0.002 ≤ 0.05 ✓
- **Verdict: replication PASSES all three declared criteria.** Fair cost under fresh
  calibration/seeds: 0.682 + 0.240 = **+0.92** (original fresh-heal frame: +0.95). Note the
  shuffled level itself is seed-sensitive (+2.21 → +4.28); the real-code level is not — the
  stable quantity is the code number, the control's magnitude varies, its direction does not.
