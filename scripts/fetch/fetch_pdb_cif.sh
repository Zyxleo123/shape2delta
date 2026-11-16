#!/bin/bash
# Download the unique PDB mmCIFs for the empirical ensembles. LOGIN NODE.
# Reads the manifest's unique_pdbs.txt; parallel, idempotent (skips existing).
set -uo pipefail
source "${S2D_ROOT:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh"

LIST="${S2D_CIF_LIST:-$S2D_FEATURES/ensemble_profiles/unique_pdbs.txt}"   # override per caller
CIFDIR="$S2D_RAW/pdb_cif"
[ -s "$LIST" ] || { echo "no $LIST (build manifest first)"; exit 1; }
mkdir -p "$CIFDIR"
n=$(wc -l < "$LIST"); echo "[cif] $n unique PDBs -> $CIFDIR @ $(date -Is)"

get_one() {
  pdb="$1"; out="$2/${pdb}.cif.gz"
  [ -s "$out" ] && exit 0
  curl -fsL --max-time 120 -o "$out.part" "https://files.rcsb.org/download/${pdb}.cif.gz" \
    && mv "$out.part" "$out" || { rm -f "$out.part"; echo "MISS $pdb"; }
}
export -f get_one
cat "$LIST" | xargs -P 16 -I{} bash -c 'get_one "$@"' _ {} "$CIFDIR"
echo "[cif] done: $(ls "$CIFDIR"/*.cif.gz 2>/dev/null | wc -l)/$n present @ $(date -Is)"
