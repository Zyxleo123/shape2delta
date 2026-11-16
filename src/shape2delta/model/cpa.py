"""A faithful, self-contained chemCPA-style compositional backbone (CPU torch).

Related-work baseline family (Deep dive, "Related work" group 2: chemCPA / CPA /
PRnet): predict the transcriptional change-from-control Δ from a *compositional*
latent that adds a drug representation (from chemistry, so it generalises to
unseen drugs — chemCPA's contribution), a dose-scaling of that drug latent, and
additive covariate embeddings. We predict Δ directly (the data is MODZ Δ, already
baseline-subtracted and unpaired), so there is no basal-expression encoder /
covariate adversary; the cell-line embedding stands in for the basal state. This
keeps the architecture recognisably CPA while making the feature on/off contrast
exact (constraint #5: "same backbone, feature on/off").

The whole point of E10: the target block enters as *one more additive
conditioning embedding* z_target, exactly as the spec says the ensemble feature
should ("an additional conditioning vector on the drug side ... alongside the
existing chemical and target representation"). Setting d_target=0 removes it;
feeding ESM / static-pocket / global-MD / static⊕MD toggles the arms.

    z = doser(logdose) * drug(chem)  +  cell_emb[cell]  +  target(target_vec)
    Δ_hat = decoder([z, time])
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:                       # torch optional at import time
    torch = None
    nn = object


def _mlp(dims, dropout=0.1, act=True):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.LayerNorm(dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class CPA(nn.Module):
    def __init__(self, n_chem, n_cells, n_genes, d_target=0,
                 d_latent=64, hidden=256, dropout=0.1):
        super().__init__()
        self.drug = _mlp([n_chem, hidden, d_latent], dropout)
        # learned per-example dose scaling of the drug latent (chemCPA "doser")
        self.doser = _mlp([1, 64, d_latent], dropout)
        self.cell = nn.Embedding(n_cells + 1, d_latent)      # last idx = unknown cell
        nn.init.normal_(self.cell.weight, std=0.1)
        self.has_target = d_target > 0
        if self.has_target:
            self.target = _mlp([d_target, hidden // 2, d_latent], dropout)
        self.dec = _mlp([d_latent + 1, hidden, hidden, n_genes], dropout)

    def forward(self, chem, logdose, cell_idx, time, target_vec):
        z = self.drug(chem) * torch.sigmoid(self.doser(logdose))
        z = z + self.cell(cell_idx)
        if self.has_target and target_vec is not None:
            z = z + self.target(target_vec)
        return self.dec(torch.cat([z, time], dim=-1))


# --------------------------------------------------------------- training ---
def _to_t(x, dtype=None):
    t = torch.as_tensor(np.ascontiguousarray(x))
    return t.to(dtype) if dtype is not None else t


def train_predict(data, tr, va, te, d_target, *, seed=0, cov_dropout=0.1,
                  d_latent=64, hidden=256, dropout=0.1, lr=1e-3, wd=1e-5,
                  batch=512, max_epochs=100, patience=8, threads=None):
    """Train one CPA on rows `tr` (early-stop on `va`), predict Δ for rows `te`.

    `data` is the dict from e10_cpa._pack (chem, logdose, cell_idx, time, Y,
    n_cells, n_genes, and the chosen `target` matrix or None). `cov_dropout`
    routes a fraction of TRAIN rows' cell index to the 'unknown' slot so that
    unseen test cells (held-out-cell split) fall back to a trained embedding.
    Returns Δ_pred[len(te), n_genes]."""
    if threads:
        torch.set_num_threads(int(threads))
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_cells = data["n_cells"]
    chem = _to_t(data["chem"], torch.float32)
    logd = _to_t(data["logdose"].reshape(-1, 1), torch.float32)
    time = _to_t(data["time"].reshape(-1, 1), torch.float32)
    cell = _to_t(data["cell_idx"], torch.long)
    Y = _to_t(data["Y"], torch.float32)
    T = _to_t(data["target"], torch.float32) if (d_target > 0 and data["target"] is not None) else None

    # test cells unseen in THIS fold's train -> unknown slot
    seen = set(np.asarray(data["cell_idx"])[tr].tolist())
    cell_te = cell[te].clone()
    for j, c in enumerate(te):
        if int(data["cell_idx"][c]) not in seen:
            cell_te[j] = n_cells
    cell = cell.clone()

    model = CPA(chem.shape[1], n_cells, data["n_genes"], d_target=d_target,
                d_latent=d_latent, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    tr = np.asarray(tr); va = np.asarray(va); te = np.asarray(te)

    def batches(idx, shuffle):
        order = torch.randperm(len(idx), generator=g).numpy() if shuffle else np.arange(len(idx))
        for s in range(0, len(idx), batch):
            yield idx[order[s:s + batch]]

    def run_batch(rows, train):
        ci = cell[rows]
        if train and cov_dropout > 0:
            drop = torch.rand(len(rows), generator=g) < cov_dropout
            ci = ci.clone(); ci[drop] = n_cells
        tv = T[rows] if T is not None else None
        pred = model(chem[rows], logd[rows], ci, time[rows], tv)
        return pred

    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        for rows in batches(tr, True):
            opt.zero_grad()
            loss = lossf(run_batch(rows, True), Y[rows])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for rows in batches(va, False):
                vloss += lossf(model(chem[rows], logd[rows], cell[rows], time[rows],
                                     T[rows] if T is not None else None), Y[rows]).item() * len(rows)
            vloss /= max(len(va), 1)
        if vloss < best - 1e-5:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(te), batch):
            rows = te[s:s + batch]
            tv = T[rows] if T is not None else None
            out.append(model(chem[rows], logd[rows], cell_te[s:s + batch], time[rows], tv).numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, data["n_genes"]), np.float32)
