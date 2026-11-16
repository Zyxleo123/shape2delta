#!/bin/bash
# LINCS L1000 Phase 1 (GSE92742) — the public, self-consistent build whose
# Level-5 MODZ matrix (the change-from-control Δ signatures) IS downloadable
# (the 2020 "beta" level-5 is paywalled at 403). RUN ON LOGIN NODE (lov4).
# Metadata is small; the Level-5 GCTX is ~21 GB. Idempotent.
set -uo pipefail
source "/zfsauton2/home/yixiz/shape2delta/slurm/preamble.sh"
DEST="$S2D_RAW/lincs_phase1"; mkdir -p "$DEST"
BASE="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl"

get(){ local url="$1" out="$2"
  if [ -s "$out" ]; then echo "[skip] $(basename "$out") ($(stat -c%s "$out")B)"; return 0; fi
  echo "[get ] $url"
  if curl -fL --retry 4 --retry-delay 10 --max-time 36000 -C - -o "$out.part" "$url"; then
    mv "$out.part" "$out"; echo "[ok  ] $(basename "$out") = $(stat -c%s "$out")B"
  else echo "[FAIL] $url (exit $?)"; return 1; fi
}

# small metadata first (fast, lets downstream code start)
for f in sig_info gene_info pert_info cell_info sig_metrics; do
  get "$BASE/GSE92742_Broad_LINCS_${f}.txt.gz" "$DEST/GSE92742_Broad_LINCS_${f}.txt.gz"
done
# the big one (resumable with -C -)
get "$BASE/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz" \
    "$DEST/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"

echo "[lincs_phase1] done $(date -Is)"; find "$DEST" -type f -printf '%12s  %p\n' | sort -k2
