"""E0 — coverage audit (the gate). Fully offline: reads only staged metadata.

Join: LINCS compounds -> single primary target -> UniProt -> ensemble-source
availability per tier. Reports targets available per tier (T0/T1/T2, with T2
split into MD vs empirical), the enzyme subset, and a drop log at every join so
a null downstream is distinguishable from thin data.

Gate rule (from the spec): if fewer than S2D_MIN_T2_TARGETS targets reach T2 via
precomputed MD, drop the MD requirement and realise T2 from empirical PDB
ensembles. Report the number either way.

Run:  python -m shape2delta.coverage
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from shape2delta import config
from shape2delta.data import targets as T
from shape2delta.data import uniprot as U
from shape2delta.data import ensembles as E


def _count_true(df: pd.DataFrame, col: str, mask=None) -> int:
    sub = df if mask is None else df[mask]
    return int(sub[col].sum())


def run() -> dict:
    config.ensure_dirs()
    atlas_id = "LINCS_L1000"

    # 1) compounds with a single primary target + measured signatures
    comp, cnt = T.build_single_target_compounds()

    # 2) gene -> UniProt accession
    statuses, accs = [], []
    for g in comp["gene_symbol"]:
        acc, st = U.map_gene(g)
        accs.append(acc or "")
        statuses.append(st)
    comp = comp.assign(accession=accs, map_status=statuses)
    map_counts = comp["map_status"].value_counts().to_dict()
    mapped = comp[comp["accession"] != ""].copy()
    cnt["compounds_gene_mapped"] = int(len(mapped))
    cnt["compounds_unmapped_gene"] = int((comp["accession"] == "").sum())

    # 3) per-target (distinct accession) tier availability
    target_accs = sorted(set(mapped["accession"]))
    cnt["distinct_targets_mapped"] = len(target_accs)
    avail = E.tier_availability(target_accs)

    # attach a representative gene symbol + compound count per target
    gene_by_acc = (mapped.groupby("accession")["gene_symbol"]
                   .agg(lambda s: sorted(set(s))[0]).to_dict())
    ncomp_by_acc = mapped.groupby("accession")["compound_id"].nunique().to_dict()
    avail["gene_symbol"] = avail["accession"].map(gene_by_acc)
    avail["n_compounds"] = avail["accession"].map(ncomp_by_acc).fillna(0).astype(int)

    enz = avail["is_enzyme"]

    def tier_counts(mask=None) -> dict:
        return {
            "T0_identity": _count_true(avail, "t0_identity", mask),
            "T1_afdb": _count_true(avail, "t1_afdb", mask),
            "T2_md_atlas": _count_true(avail, "t2_md_atlas", mask),
            "T2_md_mdcath": _count_true(avail, "t2_md_mdcath", mask),
            "T2_md_any": _count_true(avail, "t2_md_any", mask),
            "T2_empirical": _count_true(avail, "t2_empirical", mask),
            "T2_any": _count_true(avail, "t2_any", mask),
            "n_targets": int(len(avail if mask is None else avail[mask])),
        }

    targets_all = tier_counts()
    targets_enz = tier_counts(enz)

    # DRH-authoritative stratum: targets backed by >=1 DRH-sourced single-target
    # compound (cleaner than the looser LINCS-only target annotation).
    drh_accs = set(mapped.loc[mapped["target_source"] == "drh", "accession"])
    drh_mask = avail["accession"].isin(drh_accs)
    targets_drh = tier_counts(drh_mask)
    targets_drh_enz = tier_counts(drh_mask & enz)

    # 4) gate: MD vs empirical for the T2 tier
    t2_md = targets_all["T2_md_any"]
    gate_source = "MD" if t2_md >= config.MIN_T2_TARGETS else "empirical"
    effective_T2 = t2_md if gate_source == "MD" else targets_all["T2_empirical"]

    # compound-level coverage (splits are also over compounds)
    comp_avail = mapped.merge(
        avail[["accession", "t1_afdb", "t2_md_any", "t2_empirical", "t2_any", "is_enzyme"]],
        on="accession", how="left")
    comp_counts = {
        "compounds_T0": int(len(comp_avail)),
        "compounds_T1": int(comp_avail["t1_afdb"].sum()),
        "compounds_T2_md": int(comp_avail["t2_md_any"].sum()),
        "compounds_T2_empirical": int(comp_avail["t2_empirical"].sum()),
        "compounds_T2_any": int(comp_avail["t2_any"].sum()),
        "compounds_T2_enzyme": int((comp_avail["t2_any"] & comp_avail["is_enzyme"]).sum()),
    }

    result = {
        "experiment": "E0_coverage",
        "atlas": atlas_id,
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": {
            "MIN_T2_TARGETS": config.MIN_T2_TARGETS,
            "MIN_PDB_CHAINS": config.MIN_PDB_CHAINS,
        },
        "join_counts": cnt,
        "gene_map_status_counts": map_counts,
        "targets_per_tier_all": targets_all,
        "targets_per_tier_enzyme": targets_enz,
        "targets_per_tier_drh_only": targets_drh,
        "targets_per_tier_drh_only_enzyme": targets_drh_enz,
        "target_source_counts": comp["target_source"].value_counts().to_dict(),
        "compound_coverage": comp_counts,
        "gate": {
            "t2_md_targets": t2_md,
            "t2_empirical_targets": targets_all["T2_empirical"],
            "min_required": config.MIN_T2_TARGETS,
            "chosen_T2_source": gate_source,
            "effective_T2_targets": effective_T2,
            "effective_T2_enzyme_targets": (
                targets_enz["T2_md_any"] if gate_source == "MD"
                else targets_enz["T2_empirical"]),
        },
        "ensemble_source_sizes": {
            "atlas_uniprots": len(E.atlas_uniprots()),
            "mdcath_uniprots": len(E.mdcath_uniprots()),
        },
    }

    # 5) write artefacts
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS / "E0_coverage.json"
    out_json.write_text(json.dumps(result, indent=2))
    avail_out = avail[["accession", "gene_symbol", "is_enzyme", "n_compounds",
                       "t1_afdb", "n_pdb_chains", "t2_md_atlas", "t2_md_mdcath",
                       "t2_md_any", "t2_empirical", "t2_any"]].sort_values(
        ["t2_any", "n_pdb_chains"], ascending=False)
    avail_out.to_csv(config.RESULTS / "E0_target_availability.csv", index=False)
    comp.to_csv(config.RESULTS / "E0_single_target_compounds.csv", index=False)

    _update_summary(result)
    return result


def _update_summary(r: dict) -> None:
    sm = config.RESULTS / "SUMMARY.md"
    g = r["gate"]
    ta = r["targets_per_tier_all"]
    te = r["targets_per_tier_enzyme"]
    td = r["targets_per_tier_drh_only"]
    tde = r["targets_per_tier_drh_only_enzyme"]
    gate_pass = "PASS" if g["t2_md_targets"] >= g["min_required"] else "FALLBACK→empirical"
    # T2 is built best-available-per-target (MD preferred, empirical fallback),
    # so the operative T2 set is T2-any; MD-only is a cleaner robustness stratum.
    verdict = (
        f"T2 (best-available per target) reachable for {ta['T2_any']} targets "
        f"({te['T2_any']} enzymes). MD-only subset: {ta['T2_md_any']} "
        f"({te['T2_md_any']} enzymes). Gate {gate_pass} "
        f"(MD {g['t2_md_targets']} vs min {g['min_required']})."
    )
    line2 = (
        f"All single-target targets: T0/T1 {ta['T0_identity']}, T2-any {ta['T2_any']} "
        f"(enz {te['T2_any']}). DRH-authoritative stratum: T0/T1 {td['T0_identity']}, "
        f"T2-any {td['T2_any']} (enz {tde['T2_any']}). "
        f"Source: {r.get('target_source_counts', {})}."
    )
    block = [
        "## E0 — Coverage audit  ", f"_atlas: {r['atlas']} · {r['generated']}_  ",
        "", f"**Verdict.** {verdict}  ", f"{line2}  ", "",
    ]
    header = "# shape2delta — results SUMMARY\n\n"
    existing = sm.read_text() if sm.exists() else header
    if not existing.startswith("# shape2delta"):
        existing = header + existing
    # drop any previous E0 block (## E0 …) and legacy one-line E0 bullets
    lines = existing.splitlines()
    kept, skip = [], False
    for ln in lines:
        if ln.startswith("## E0 "):
            skip = True
            continue
        if skip and ln.startswith("## ") and not ln.startswith("## E0 "):
            skip = False
        if ln.lstrip().startswith("- **E0"):
            continue  # strip legacy bullet from an earlier run
        if not skip:
            kept.append(ln)
    # drop orphan non-blank lines between the H1 and the first '## ' section
    cleaned, seen_section = [], False
    for ln in kept:
        if ln.startswith("## "):
            seen_section = True
        if (not seen_section) and (not ln.startswith("# ")) and ln.strip():
            continue
        cleaned.append(ln)
    sm.write_text("\n".join(cleaned).rstrip() + "\n\n" + "\n".join(block) + "\n")


if __name__ == "__main__":
    res = run()
    print(json.dumps({k: res[k] for k in
                      ("targets_per_tier_all", "targets_per_tier_enzyme",
                       "gate", "ensemble_source_sizes")}, indent=2))
