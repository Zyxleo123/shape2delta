# shape2delta

**Does a drug target's *conformational ensemble* carry transferable signal for
transcriptional drug-response prediction?**

A single offline harness that answers that question with a controlled comparison.
The target protein is represented at three strictly nested tiers:

| tier | what it adds | source |
|------|--------------|--------|
| **T0** identity | one-hot target + ESM-2 sequence embedding | UniProt sequence / ESM-2 |
| **T1** static structure | pLDDT/PAE summaries, fpocket descriptors, R_g, SS fractions | AlphaFold DB model |
| **T2** ensemble | per-residue RMSF, state-population summaries, pocket-volume variance, cryptic-pocket descriptors | precomputed MD (ATLAS / mdCATH / GPCRmd) → else ≥5-chain empirical PDB ensemble → else BioEmu/AlphaFlow |

**T1 ⊂ T2 by construction.** Every tier reduces to a fixed-length vector with
identical dimensionality handling, so the only thing varying between arms is tier
membership. **The headline number is always the T2−T1 gap, never absolute accuracy.**

Prediction target: **change-from-control Δ** at the pseudobulk level. Metrics:
Δ-Pearson and differential-gene recovery (top-100 overlap). Evaluation is on
**held-out-target, held-out-compound-scaffold, and held-out-cell-context** splits
with ≥5 folds — never random example splits. Every comparison includes a
**shuffled-target permutation arm**. Gaps are reported as a bootstrap mean with a
95% CI over folds; a gap is treated as real only if the CI excludes zero in the
same direction on ≥2 split types.

## Data
- **Perturbation atlas (primary): LINCS L1000** level-5 consensus signatures
  (CLUE 2020 beta: `compoundinfo_beta`, `siginfo_beta`, `geneinfo_beta`). Broadest
  compound/target coverage → enough targets to form target/scaffold/context folds.
- **Perturbation atlas (secondary): sci-Plex3** (GEO GSE139944) — true scRNA-seq,
  used where single-cell distribution matching is required (E4).
- **Targets:** Drug Repurposing Hub (authoritative single-vs-multi target), ChEMBL.
- **Structure/ensemble:** AlphaFold DB (T1); ATLAS + mdCATH + GPCRmd (MD), SIFTS
  `pdb_chain_uniprot` for empirical ensembles and PDB↔UniProt mapping.
- **Specificity control:** Replogle K562 CRISPRi (E5) — knockdown removes the
  protein, so the prediction is *no* ensemble gain.

Raw data is staged on scratch (`$S2D_DATA`); the repo holds only code + results.

## Experiments (run in order; each independently reportable)
| id | what | compute | status |
|----|------|---------|--------|
| **E0** | coverage audit — targets per tier (the gate) | CPU | **done** — 688 targets; T2-any 546 (297 enz); MD-only 73/51; DRH stratum 409/217; gate PASS |
| **E1** | feature-only transferability (partial Mantel), no model | CPU | **done — null** (partial-r 0.008, p=0.16) |
| **E2** | generalization-gap baseline (random vs held-out splits) | CPU | **done** — gap real (drop 0.09–0.15 Δρ, CI excl 0) |
| **E3** | core ablation T0/T1/T2 with ridge on Δ + permutation | CPU | **done — no practical gain** (T2−T1 ≤ +0.0008 Δρ) |
| **E4** | full-backbone ablation (flow/transport), MD loss intact | GPU | deferred (budget; E1/E3 null → low priority) |
| **E5** | specificity negative control on Replogle CRISPRi | CPU | **done** — both null; no identity leakage |
| **E6** | mechanism-level sanity scatter (flexibility vs response) | CPU | **done — null** (partial 0.007) |
| **E7** | pocket-centric *matched* ablation (ligand-defined pocket; T1 = AF pocket, T2 = T1 ⊕ its motion over PDB conformers): ridge + held-out-target kNN + partial Mantel | CPU | **done — null** (282 targets, all classes; ridge T2−T1 ≤ 0; kNN −0.021 [−0.035,−0.007] = motion *hurts*; Mantel r −0.001, p 0.96) |
| **E8** | real MD ensembles (ATLAS + mdCATH frames) as the T2 source, same pocket/T1/summariser; pure-Python .xtc decoder (RMSF r≥0.999 vs ATLAS) | CPU | **done — null** (20 targets w/ MD covering the pocket; nothing clears the 0.01 floor; MD & PDB flexibility agree but neither predicts Δ) |
| **E9** | single-cell readout (sci-Plex3): matched pocket tiers on pseudobulk Δ **and** on single-cell distribution readouts (E-dist, responder-frac, dispersion, bimodality, viability) | CPU | **done — null** (34 targets, 2247 conditions; ridge T2−T1 ≤ 0; kNN −0.001 [−0.044,+0.040]; Mantel r −0.014; single-cell screen nothing survives BH) |
| **E10** | MD as an *extra input* to a real related-work backbone (faithful chemCPA-style compositional predictor): global whole-protein MD vector enters as one more conditioning embedding; arms OFF/ESM/STATIC/MD/STATIC+MD/MD-permuted, same held-out splits | CPU | **done — null** (66 MD targets / 2581 cond, 3 seeds; (STATIC+MD)−STATIC max +0.003 Δρ, MD−STATIC max +0.002, both < 0.01 floor; MD ≈ permuted-MD; OFF ≥ every conditioned arm on held-out targets) |

**Bottom line:** ensemble (T2) features add no transferable signal over identity/
static here. The headline null (E1/E3/E5/E6; E2 confirms the gap exists) survives
every lever that could have rescued it: the spec's matched pocket descriptor (E7),
real molecular dynamics instead of PDB conformers (E8), single-cell readout
instead of bulk (E9), and — the last lever — feeding MD as an extra input to an
*actual related-work backbone* (a chemCPA-style compositional predictor) over the
broadest MD set we have (E10) — all null. In E10 the MD block is indistinguishable
from a target-shuffled permutation, and the no-target backbone is as strong as any
conditioned arm on held-out targets. Figures in `results/figures/`. See
`results/SUMMARY.md` for the honest headline + caveats.

## Layout
```
slurm/preamble.sh      shared config — ALL defaults live here, sourced by every job
slurm/submit.sh        submitter: writes a timestamped env-file, passes its PATH as
                       a positional arg (never --export); scrubs SLURM_*
slurm/e*.sbatch        one job per stage; CPU vs GPU split by resource
scripts/fetch/         LOGIN-NODE downloads (compute nodes may lack internet)
src/shape2delta/       config, data loaders, feature builders, experiment code
results/               *.json per experiment + SUMMARY.md (two-line verdicts)
```

## Running
```bash
# one-time: stage metadata on the login node (has internet)
bash scripts/fetch/fetch_e0.sh
# submit a stage (from lov4; prints the exact sbatch line)
bash slurm/submit.sh e0_coverage.sbatch
# chain a dependent stage after a job id
bash slurm/submit.sh e3_ablation.sbatch <jobid> ATLAS=LINCS_L1000
```
Configuration is by environment variable with defaults in `slurm/preamble.sh`;
override per submission with `KEY=VAL` args to `submit.sh` (written into the
env-file, sourced by the job). Jobs print `hostname`, `nvidia-smi -L` and their
resource request at startup. Figures are redrawable from the JSON/CSV artefacts
without recomputation.
