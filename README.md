# explainable-ai

**📄 Read the paper: [How Much of a Transformer Is Code?](https://keithagroves.github.io/how-much-transformer-is-code/)**

Testing one idea: **how much of a neural network is "just rules"** — computation that
ordinary, human-readable code could reproduce — and where that stops.

> Note: the large data artifacts (corpora, Gutenberg books, twin datasets, `*.npy` caches,
> `*.pt` weights) are gitignored to keep the repo lean; scripts regenerate or re-download them.

The work runs in two chapters. Chapter 1 imitates a small language model from the outside
with hand-authored rulebooks; chapter 2 opens real transformer weights (Qwen3-0.6B,
Pythia-410M/160M) and replaces parts of them with code, healing the network around the
prosthetics. Full narrative in [`writeup/`](writeup/).

## Read first

- **[writeup/article.html](writeup/article.html)** — concise, abstract-first paper (source for the
  published page at [`docs/index.html`](docs/index.html); edit the former and re-wrap into the latter).
- **[writeup/rules_writeup.html](writeup/rules_writeup.html)** — the full six-act field report.
- **[writeup/legible_model_built.html](writeup/legible_model_built.html)** — the Legible Model
  demo (rule-only Q&A + text generator, self-contained; open in a browser).
- **[writeup/report.html](writeup/report.html)** — the chapter-1 sentiment field note.
- **[plan.md](plan.md)** — architecture ladder and early planning notes.

## Layout

```
README.md   plan.md
writeup/    the five deliverable HTML pages
src/        all code + the data it reads (kept together on purpose — see below)
```

## Code

Everything lives in **`src/`** and is run from there (`cd src && python3 …`). One experiment
per file, flat, each importable by the next: later scripts do `import replace_all as RA` and
read data with bare relative paths like `open("ministral_corpus.txt")`, so the scripts and
their data must share one directory — that is why `src/` holds both, rather than splitting
code and data into separate folders. Roughly in creation/narrative order:

**Chapter 1 · black-box imitation**
- Sentiment surrogate: `corpus.py` `label.py` `embed.py` `probe.py` `surrogate.py` `model.py` `abstain.py` `sweep.py`
- Next-token rulebooks + negatives: `nexttoken.py` `ngram_lm.py` `rule_program.py` `extract_rules.py` `generalize.py` `compare_predictors.py` `cache_lm.py` `dialogue_lm.py` `smooth_lm.py` `class_lm.py` `vector_lm.py`
- Two-spaces / distributional: `dist_lm.py` `sae.py` `compose_*.py`
- "Nesses" / arbitrary scales: `nesses.py` `ness_validate.py` `scales.py` `build_lexicon.py` `tone_rerank.py`
- Structured-input turn: `gen_structured.py` `struct_eval.py` `schema_lm.py` `slot_lm.py` `mine_rules.py` `template_automaton.py` `nlg_final.py`
- Q&A distillation + demo: `qa_distill.py` `demo.py` `export_demo.py`
- Data source: `gen_ministral2.py` (`ministral_corpus.txt`), `winograd.py` `spec_test.py`

**Chapter 2 · white-box surgery** (all on real weights, PyTorch hooks, fp32 on MPS)
- Find + prove the circuit: `induction_hunt.py` `ablate.py` `natural_rule.py` `veto_hunt.py`
- Prosthesis ladder (substitute the induction circuit with code): `prosthesis.py`…`prosthesis5.py` `ceiling.py` `fuzzy.py`
- Replacement frontier + healing: `replace_all.py` `replace_all2.py` `replace_rich.py` `mlp_prosthesis.py` `heal.py` `heal2.py` `heal_mlp.py` `heal_combined.py` `heal_control.py` `heal_posonly.py`
- Selection-strategy sensitivity (is cheapest-solo-first optimal?): `select_strategy.py` (in-context marginal costs) `select_verify.py` (measured swap curve)
- Reviewer follow-ups: `disambig.py` (heads/MLPs/both damage decomposition — resolves the +1.61/+0.68 basis) `heal_eval.py` (pooled healed eval) `heal_holdout.py` (out-of-sample pooled heal: +0.64 nats [+0.57,+0.71] on 8 chunks held out of the LUT, norm heal, and early-stopping — makes the +0.67 headline measured, not indicative) `heal_shuffle.py` (**shuffled-code heal control**: heal real vs position-scrambled code under identical protocol — real +0.70 nats vs shuffled +2.21 [+2.08,+2.33] ≈ zero-ablation, so the heal recovers function, not distribution) `joint_posonly.py` (positional-only control on the joint-search set)
- Robustness + generality: `ceiling_suite.py` `ceiling_domains.py` (now with 95% cluster-bootstrap CIs on every ceiling %) `ceiling_vocab.py` (entity-vs-random with CIs: Qwen +21% [+10,+43], Pythia +8% [+3,+14], both clear zero) `wikitext_eval.py` `pythia_session2.py`
- Scale replication (Qwen3-4B, bf16): `arch_scale.py` (induction + MLP U-map) `arch_heads.py` (per-head thin-tail) `ceiling_4b.py` (domain ceiling)
- Causal twin attempts: `prep_twins.py` `train_twin.py` `twin_ceiling.py` `pythia_twin.py` `atrophy_trajectory.py`
- Entity probe (what the residue is): `probe_rank.py` `probe_rank2.py` `probe_substitute.py`
- Query-edit causal probe (is the residue direction *wired in*): `qk_recon.py` (Qwen attention-reconstruction validator) `probe_queryedit.py` (Qwen: gap +0.49 [+0.35,+0.64]). Cross-family port: `qk_recon_neox.py` (GPT-NeoX validator — fused QKV, partial rotary, LayerNorm) `probe_queryedit_neox.py` (Pythia: stronger, gap +0.82 [+0.46,+1.16]) — the one positive residue claim now replicates on two families.
- Coreference/primacy rung (is the selection a nameable discourse rule): `probe_coref.py` (Qwen + rank-1 reference) `probe_coref_x.py` (cross-family, Pythia). `probe_primacy_control.py` — the sink control that **retracted** the primacy result (it was the attention sink). `probe_coref2.py` — sink-controlled test of every readable selection rule (recency/coreference/salience); conclusive negative: the which-entity selection is smooth. `probe_syntax.py` (clause-subject via spaCy) and `probe_coref_proper.py` (neural coreference via fastcoref) — the serious linguistic rules through the sink gate; both fail cross-family, hardening "un-nameable". `probe_softrule.py` — distributional feature-bundle probe: a soft weighting of nameable features (and even the learned rank-1 direction) fails to beat recency under sink control — hardens "un-nameable" AND retracts the old low-rank "0.61 reconstruction" as a third sink artifact. `probe_softrule_q.py` — adds the reviewer's **query-conditional** rule ("attend to the entity that co-occurred with the current query token"); active on a third of cases but adds nothing over recency (mass-overlap 0.333 vs 0.331), so the residue resists a query-side vocabulary too — reframed as a bounded negative (no rule *in this vocabulary, under this soft-model class, at this scale*).

Files with suffixes like `_s12`, `_192`, `_oproj` are seed/config variants of the run they name.

## Data & checkpoints (also in `src/`)

Corpora are `*_corpus.txt` (chief: `ministral_corpus.txt`, `structured_corpus.txt`); `pg*.txt`
are raw Project Gutenberg source. Cached embeddings are `*_emb.npy` / `word_cache.npy`.
Fitted artifacts are `*.pt` (head scores, templates, healed norms, solo costs). Most `.pt`
files are cheap to regenerate by rerunning the script that saved them.

## Status

Concluded as a research artifact. Findings, negatives, and the honest significance assessment
are in the write-ups. The running project record lives in the agent memory file, not here.
