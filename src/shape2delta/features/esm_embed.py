"""T0 identity features: ESM-2 sequence embeddings (mean-pooled).

esm2_t30_150M on CPU (truncate to 1022 residues, mean-pool the final layer) →
a 640-d embedding per target. Weights are pre-downloaded on the login node
(download_weights) so the compute job runs offline.

  login node:  python -c "from shape2delta.features.esm_embed import download_weights; download_weights()"
  SLURM (CPU): python -m shape2delta.features.esm_embed
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shape2delta import config

MODEL = "esm2_t12_35M_UR50D"   # 480-d; fast on CPU (T0 identity + E1 ESM control)
LAYER = 12
EMB_DIM = 480
OUT = config.FEATURES / "esm"
MAXLEN = 1022


def _load_model():
    import esm
    import torch
    fn = getattr(esm.pretrained, MODEL)
    model, alphabet = fn()
    model.eval()
    torch.set_grad_enabled(False)
    return model, alphabet


def download_weights() -> str:
    _load_model()  # triggers the hub download into ~/.cache/torch/hub
    return "weights cached"


def read_fastas() -> dict[str, str]:
    seqs = {}
    d = config.RAW_UNIPROT / "fasta"
    for fa in sorted(d.glob("*.fasta")):
        acc = fa.stem
        lines = fa.read_text().splitlines()
        seq = "".join(l.strip() for l in lines if l and not l.startswith(">"))
        if seq:
            seqs[acc] = seq
    return seqs


def build(batch_tokens: int = 8000) -> dict:
    import torch
    model, alphabet = _load_model()
    bc = alphabet.get_batch_converter()
    seqs = read_fastas()
    accs = sorted(seqs)
    items = [(a, seqs[a][:MAXLEN]) for a in accs]
    # sort by length, greedily pack batches under a token budget
    items.sort(key=lambda x: len(x[1]))
    embs: dict[str, np.ndarray] = {}
    i = 0
    while i < len(items):
        j = i
        L = len(items[i][1])
        # grow batch while token budget allows
        while j < len(items) and (j - i + 1) * max(L, len(items[j][1])) <= batch_tokens:
            L = max(L, len(items[j][1])); j += 1
        batch = items[i:max(j, i + 1)]
        labels, strs, toks = bc(batch)
        out = model(toks, repr_layers=[LAYER])["representations"][LAYER]
        for k, (acc, s) in enumerate(batch):
            # mean over real residues (positions 1..len(s)); skip BOS/EOS/pad
            rep = out[k, 1:len(s) + 1].mean(0).numpy().astype(np.float32)
            embs[acc] = rep
        i = max(j, i + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    acc_list = sorted(embs)
    mat = np.stack([embs[a] for a in acc_list]) if acc_list else np.zeros((0, EMB_DIM), np.float32)
    np.save(OUT / f"{MODEL}.npy", mat)
    (OUT / f"{MODEL}.accs.json").write_text(json.dumps(acc_list))
    summ = {"model": MODEL, "n_targets": len(acc_list), "dim": int(mat.shape[1]) if len(mat) else 0}
    (OUT / "build_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def load() -> tuple[np.ndarray, list[str]]:
    mat = np.load(OUT / f"{MODEL}.npy")
    accs = json.loads((OUT / f"{MODEL}.accs.json").read_text())
    return mat, accs


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
