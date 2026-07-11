"""Emit one reproducible, concrete replacement program for the paper.

The attention example is a representative member of the headline 160-head set.
The MLP example is one real frequent-token row from the cheapest substituted
layer, decomposed into the paper's rank-64 base plus exact-token correction.
"""
import collections
import json
import torch
import replace_all as RA

model, DEV, tokz = RA.model, RA.DEV, RA.tokz

# Representative compact attention program, selected once from the headline set.
HEAD = (5, 15)
weights = torch.load("rich_templates.pt")[HEAD]
nat_cost = torch.load("rich_solo_costs.pt")[HEAD]
head_terms = [
    {"instruction": name, "coefficient": float(value)}
    for name, value in sorted(weights.items(), key=lambda item: -item[1])
    if value > 1e-4
]

# Rebuild the calibration LUT for one actual substituted MLP layer.
mlp_costs = torch.load("mlp_solo_costs.pt")["costs"]
layer = sorted(mlp_costs, key=lambda l: mlp_costs[l])[0]
raw = open("ministral_corpus.txt").read()
starts = [o for o in range(20000, min(len(raw)-10000, 1000000), 40000)
          if not 185000 <= o <= 215000][:24]
chunks = [tokz.encode(raw[o:o+8000])[:600] for o in starts]

captured = {}
hook = model.model.layers[layer].mlp.register_forward_hook(
    lambda mod, inp, out: captured.__setitem__("out", out[0].detach().float().cpu()))
sums, counts, freq = {}, collections.Counter(), collections.Counter()
with torch.no_grad():
    for seq in chunks:
        model(torch.tensor([seq]).to(DEV)); out = captured["out"]
        freq.update(seq)
        for i, token in enumerate(seq):
            if token in sums:
                sums[token].add_(out[i]); counts[token] += 1
            else:
                sums[token] = out[i].clone(); counts[token] = 1
hook.remove()
tokens = list(sums)
table = torch.stack([sums[t] / counts[t] for t in tokens])

# Choose the most frequent alphabetic token so the row is recognizable.
token = next(t for t, _ in freq.most_common()
             if tokz.decode([t]).strip().isalpha())
row_index = tokens.index(token); exact = table[row_index]
mean = table.mean(0, keepdim=True)
U, S, Vh = torch.linalg.svd(table - mean, full_matrices=False)
rank = 64
base = mean[0] + (U[row_index, :rank] * S[:rank]) @ Vh[:rank]
correction = exact - base
rel_error = correction.norm() / exact.norm().clamp(min=1e-12)
top500 = {t for t, _ in freq.most_common(500)}

result = {
    "model": "Qwen/Qwen3-0.6B",
    "attention": {
        "layer": HEAD[0],
        "head": HEAD[1],
        "solo_substitution_damage_nats": float(nat_cost),
        "coefficient_sum_before_row_normalization": float(sum(weights.values())),
        "terms": head_terms,
    },
    "mlp": {
        "layer": layer,
        "token_id": token,
        "token_text": tokz.decode([token]),
        "calibration_occurrences": counts[token],
        "hidden_dimensions": exact.numel(),
        "exact_row_l2_norm": float(exact.norm()),
        "rank64_relative_row_error_before_exception": float(rel_error),
        "is_top500_exception": token in top500,
        "exact_row_first_8": [float(x) for x in exact[:8]],
        "rank64_base_first_8": [float(x) for x in base[:8]],
        "exception_correction_first_8": [float(x) for x in correction[:8]],
    },
}
with open("worked_example.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
