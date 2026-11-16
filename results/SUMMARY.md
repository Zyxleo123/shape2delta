# shape2delta — results SUMMARY

## Headline (honest)
**The target's conformational ensemble carries no transferable signal for LINCS
transcriptional drug-response prediction, in this setup.** The T2−T1 gap is ~0
three independent ways: in raw feature space (E1 partial Mantel r=0.008, p=0.16),
in the ridge ablation (E3 gap +0.0005 Δρ — statistically detectable only because
n=13.7k, practically negligible, and the shuffled-target arm is within noise), and
in the mechanism scatter (E6 partial r=0.007). E2 confirms the generalization gap
the ensemble was meant to repair is real. E5 (knockdown) is null too, which rules
out identity leakage. This is a clean, well-powered negative result (337–349
targets), not thin data.

**The null survives every lever we had left — matched pocket features (E7),
real MD (E8), and single-cell readout (E9).** Three independent ways the earlier
run could have been wrong, each closed:

- **E7 — matched pocket descriptor.** E1–E6 used 9 global CA-RMSF stats with T1
  from a different extractor. E7 uses the spec's pocket-centric T2 and ONE
  extractor for both tiers: drug pocket = ligand contacts in holo PDB chains
  (271/282 targets; AF-fpocket fallback for 11); T1 = that pocket's descriptor on
  the AlphaFold model; T2 = T1 ⊕ the same descriptor's motion over ≤30 PDB
  conformers (pocket opening, pocket RMSF, state count/entropy, pairwise RMSD,
  cryptic-pocket flag). **282 targets** (all protein classes; the fpocket wrapper
  bug that zeroed every druggability column is fixed — 4137/4137 conformer runs
  OK). Ridge T2−T1 ≤0 on every split; held-out-target kNN centred gap −0.021
  [−0.035,−0.007] (motion *hurts*); partial Mantel r=−0.001 (p=0.96).
- **E8 — real molecular dynamics.** Same pocket/T1/summariser, conformers from
  ATLAS + mdCATH trajectories (a from-scratch pure-Python .xtc decoder, validated
  against ATLAS's own RMSF, r≥0.999). Only **20 LINCS targets** have an MD entry
  that resolves ≥60% of the drug pocket (mdCATH's CATH domains usually miss it),
  so this arm is underpowered. No test clears the 0.01 floor: ridge MD−T1 on
  held-out target +0.007 (excludes 0 but sub-floor, and the shuffled-target
  control is within noise); kNN CIs all include 0. MD and PDB flexibility agree
  with each other (fp-druggability-CV ρ=0.54) yet neither predicts Δ.
- **E9 — single-cell readout (sci-Plex3).** 2247 conditions, 34 drug targets, the
  matched pocket tiers scored on single-cell pseudobulk Δ (2722 genes) AND on
  distribution-level readouts LINCS cannot give (E-distance to vehicle, responder
  fraction, dispersion, bimodality, viability). Ridge T2−T1 ≤0; kNN centred
  −0.001 [−0.044,+0.040]; Mantel r=−0.014 (p=0.89). The single-cell screen (56
  motion-feature × readout tests, correctly aggregated to the **target** level to
  avoid pseudoreplication) has nothing survive BH (best raw hit pocket-volume-
  opening → responder-fraction ρ=0.55 at p=0.005, q=0.57). Interestingly static
  structure does add a little over identity here (kNN T1−T0 +0.059, CI excl 0) —
  but motion adds nothing on top of static.

**Scope caveats (remaining):** MD coverage of druggable targets is thin (20) — a
larger MD-with-pocket set is the one lever not yet well-powered; sci-Plex3 is 3
cell lines / 188 compounds (39 single-target); ESM-2 35M; the E4 nonlinear flow
backbone is not run (low priority — no linear or neighbour signal for it to find).

## E0 — Coverage audit  
_atlas: LINCS_L1000 · 2026-11-15T21:44:04.870551+00:00_  

**Verdict.** T2 (best-available per target) reachable for 546 targets (297 enzymes). MD-only subset: 73 (51 enzymes). Gate PASS (MD 73 vs min 40).  
All single-target targets: T0/T1 688, T2-any 546 (enz 297). DRH-authoritative stratum: T0/T1 502, T2-any 409 (enz 217). Source: {'drh': 1840, 'lincs': 1137}.

## E2 — Generalization gap  
_atlas: LINCS_L1000 · n=23380 conditions · 2026-11-15T22:03:14.595306+00:00_  

**Verdict.** Baseline Δρ: random 0.273 → target 0.181, scaffold 0.119, cell 0.182. target Δρ drop +0.092 [+0.084,+0.102]*; scaffold Δρ drop +0.154 [+0.149,+0.158]*; cell Δρ drop +0.091 [+0.086,+0.096]*  (* = 95% CI excludes 0)

## E6 — Mechanism scatter  
_atlas: LINCS_L1000 · 2026-11-15T22:56:56.133801+00:00_  

**Verdict.** Flexibility (RMSF) vs #DE genes: Spearman +0.011, partial(|dose,size) +0.007; vs signature entropy partial +0.026 (349 targets).

## E1 — Feature-only Mantel  
_atlas: LINCS_L1000 · 2026-11-15T22:57:20.137248+00:00_  

**Verdict.** Ensemble-sim vs Δ-sim partial-r (|ESM) = +0.008 (Mantel p=0.159, 337 targets); simple r=+0.010, ESM-only r=+0.043, static-struct partial r=+0.011. → **null**.

## E3 — Core ablation (T0/T1/T2)  
_atlas: LINCS_L1000 · n=13765 cond, 337 targets · 2026-11-15T23:02:36.711953+00:00_  

**Verdict.** T2−T1 Δρ gap: target -0.000[-0.001,+0.000]; scaffold +0.001[+0.000,+0.001]*; cell +0.000[+0.000,+0.001]*; random +0.000[+0.000,+0.000]* (* = CI excl 0). → **No practical ensemble gain**: the largest gap is +0.0008 Δρ (<0.5% of baseline); CIs exclude 0 only because n=13.7k. T0≈T1≈T2 everywhere.

## E5 — Specificity control (CRISPRi)  
_2026-11-15T23:04:52.199225+00:00_  

**Verdict.** T2−T1 gap: chemical -0.001[-0.004,+0.002] (337 targets) vs CRISPRi -0.001[-0.009,+0.006] (43 targets) — both null. There is no chemical gap for the knockdown to be specific against; the value of this arm is that CRISPRi shows no spurious gap either, ruling out target-identity leakage as a confound. (CRISPRi arm thin: only 43 Replogle genes overlap the feature set.)

## E9 - Single-cell readout (sci-Plex3)  
_atlas: sci-Plex3 · 2026-11-16T01:00:05.010456+00:00_  

**Verdict.** Single-cell atlas: 34 targets w/ pocket feats (23 PDB-ensemble, 3 MD), 594 conditions. (a) ridge T2-T1 Delta-rho: random -0.000[-0.000,-0.000]*; target -0.001[-0.003,+0.001]; scaffold -0.001[-0.002,+0.000]; cell +0.000[-0.000,+0.000]. (b) held-out-target kNN centred T2-T1 -0.001[-0.044,+0.040], T2-perm -0.021[-0.074,+0.029] (34 targets). (c) partial Mantel motion vs Delta | ESM+static r=-0.014 (p=0.887). (d) best single-cell assoc e_fp_vol_open->frac_responders partial-rho +0.55 (q=0.57). -> **null** (floor 0.01; * = CI excl 0).

## E8 — Real MD ensembles (ATLAS/mdCATH)  
_atlas: LINCS_L1000 · 2026-11-16T01:02:28.448650+00:00_  

**Verdict.** MD ensembles (ATLAS/mdCATH frames, same pocket/T1/summariser as E7) for 20 targets ({'atlas': 11, 'mdcath': 9}; median pocket cover 1.00). (a) ridge T2_MD−T1 Δρ: target +0.007[+0.002,+0.011]*; scaffold +0.000[+0.000,+0.001]*; cell +0.000[-0.000,+0.000]; random +0.000[-0.000,+0.000]. (b) held-out-target kNN centred T2_MD−T1 +0.051[-0.010,+0.118], T2_MD−T2_PDB -0.016[-0.070,+0.040], T2_MD−perm +0.004[-0.065,+0.071]. (c) partial Mantel MD-motion vs Δ | ESM+static r=+0.034 (p=0.746); PDB-motion on the same targets r=+0.053. → **null** (floor 0.01; * = CI excl 0).

## E7 — Pocket-centric matched ablation  
_atlas: LINCS_L1000 · 2026-11-16T01:04:53.308436+00:00_  

**Verdict.** Matched pocket tiers (282 targets, 184 enzymes; median 11 conformers; T1 = AF pocket descriptor, T2 = T1 ⊕ its motion over experimental conformers). (a) ridge T2−T1 Δρ: target -0.004[-0.006,-0.003]*; scaffold +0.000[-0.000,+0.001]; cell +0.000[+0.000,+0.001]*; random -0.000[-0.000,-0.000]*. (b) held-out-target kNN, centred Δρ T2−T1 -0.021[-0.035,-0.007]*, T2−perm +0.002[-0.016,+0.019] (273 targets). (c) partial Mantel motion vs Δ | ESM+static r=-0.001 (p=0.964). → **null** (effect floor 0.01; * = CI excl 0).

## E10 — MD as extra input to a chemCPA-style baseline  
_atlas: LINCS_L1000 · 2026-11-16T04:49:19.534160+00:00_  

**Verdict.** chemCPA-style backbone on LINCS, MD as an extra conditioning input, over 66 MD targets / 2581 conditions ({'mdcath': 48, 'atlas': 18}). Absolute Δρ (held-out target/scaffold/cell/random): OFF 0.114/0.152/0.263/0.356, STATIC+MD 0.109/0.164/0.262/0.336. (STATIC+MD)−STATIC Δρ: target +0.003[-0.005,+0.011]; scaffold -0.002[-0.008,+0.004]; cell -0.001[-0.002,+0.000]; random -0.009[-0.017,-0.000]*. MD−STATIC max +0.0021; (STATIC+MD)−STATIC max +0.0030. → **MD adds nothing over static** (floor 0.01; * = CI excl 0).  

