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

**E7 closes the main methodological gap and is also null.** E1–E6 used 9 global
CA-RMSF stats with T1 from a different extractor. E7 re-ran the test with the
spec's pocket-centric T2 and a *matched* extractor: the drug pocket is defined
from ligand contacts in holo PDB chains (163/169 targets; AF fpocket fallback for
6), T1 = that pocket's descriptor on the AlphaFold model (geometry, fpocket
volume/druggability, pLDDT), T2 = T1 ⊕ the same descriptor's motion over ≤30
experimental conformers (median 12): pocket opening, pocket RMSF, state
count/entropy, pairwise RMSD, cryptic-pocket flag. 169 enzyme targets, 2568
conformers, all fpocket runs OK. Result: ridge T2−T1 is ≤0 on every split
(held-out target −0.004, i.e. motion slightly *hurts*); held-out-target kNN
centred gap −0.000 [−0.018, +0.017]; partial Mantel motion-vs-Δ | ESM+static
r = −0.006 (p = 0.83). Motion features alone give kNN centred Δρ 0.011 (CI
includes 0). No per-feature association with response strength survives BH
(best q = 0.10, the `*_open` pocket-opening features, mildly count-confounded:
ρ = 0.12–0.24 with conformer count).

**Scope caveats (remaining):** ensembles are empirical PDB conformers, not MD /
enhanced sampling, and are mostly holo (the AF-vs-experiment cryptic flag is
partly induced fit; the apo-vs-holo variant exists for only 93 targets); LINCS
bulk consensus (not single-cell); ESM-2 35M; E7 power is lower than E3 (164
targets, 5.7k conditions; kNN rules out centred gains above ≈ +0.017 Δρ); E4
flow backbone not run.

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

## E7 — Pocket-centric matched ablation  
_atlas: LINCS_L1000 · 2026-11-15T23:52:57.368089+00:00_  

**Verdict.** Matched pocket tiers (169 enzyme targets, median 12 conformers; T1 = AF pocket descriptor, T2 = T1 ⊕ its motion over experimental conformers). (a) ridge T2−T1 Δρ: target -0.004[-0.008,-0.001]*; scaffold -0.001[-0.002,-0.000]*; cell +0.000[-0.000,+0.000]; random -0.000[-0.000,-0.000]*. (b) held-out-target kNN, centred Δρ T2−T1 -0.000[-0.018,+0.017], T2−perm +0.014[-0.006,+0.036] (164 targets). (c) partial Mantel motion vs Δ | ESM+static r=-0.006 (p=0.826). → **null** (effect floor 0.01; * = CI excl 0).  

