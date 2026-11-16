#!/bin/bash
# E0 metadata fetch — RUN ON THE LOGIN NODE (lov4): compute nodes may lack
# internet, so all network I/O is staged here into $S2D_RAW. Idempotent: skips
# any file already present and non-empty. Writes a size manifest for the audit.
#
#   bash scripts/fetch/fetch_e0.sh
#
# Downloads only small/medium METADATA needed for the coverage audit. The big
# expression matrices (sci-Plex3 3.1GB, LINCS GCTX) are fetched by later stages.
set -uo pipefail
source "${S2D_ROOT:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh"

MANIFEST="$S2D_RAW/fetch_e0_manifest.tsv"
: > "$MANIFEST"
log(){ printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$MANIFEST"; }

# get <url> <dest> : skip if dest exists & >1KB; else curl with retries
get() {
  local url="$1" dest="$2"
  if [ -s "$dest" ] && [ "$(stat -c%s "$dest")" -gt 1024 ]; then
    log "SKIP" "$(stat -c%s "$dest")" "$dest"; return 0
  fi
  echo "[fetch] $url -> $dest"
  if curl -fL --retry 3 --retry-delay 5 --max-time 1800 -o "$dest.part" "$url"; then
    mv "$dest.part" "$dest"; log "OK" "$(stat -c%s "$dest")" "$dest"
  else
    rm -f "$dest.part"; log "FAIL" "0" "$dest ($url)"; return 1
  fi
}

echo "=========== E0 fetch @ $(date -Is) on $(hostname) ==========="

# --- 1. LINCS L1000 (CLUE 2020 beta) -----------------------------------------
get "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt" \
    "$S2D_RAW/lincs/compoundinfo_beta.txt"
get "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt" \
    "$S2D_RAW/lincs/siginfo_beta.txt"
get "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt" \
    "$S2D_RAW/lincs/geneinfo_beta.txt"

# --- 2. Drug Repurposing Hub (compound -> MOA + target) ----------------------
get "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt" \
    "$S2D_RAW/drh/repurposing_drugs_20200324.txt"
get "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt" \
    "$S2D_RAW/drh/repurposing_samples_20200324.txt"

# --- 3. UniProt human reviewed: gene symbol -> accession (+synonyms,+PDB) -----
UNIPROT_URL="https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&fields=accession,id,gene_primary,gene_names,protein_name,xref_pdb,length&format=tsv&compressed=true"
get "$UNIPROT_URL" "$S2D_RAW/uniprot/human_reviewed.tsv.gz"

# --- 4. SIFTS: PDB chain -> UniProt (empirical ensemble chain counts) ---------
get "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_uniprot.tsv.gz" \
    "$S2D_RAW/pdb_sifts/pdb_chain_uniprot.tsv.gz"

# --- 5. mdCATH domain list (MD ensemble source) ------------------------------
get "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/mdCATH_domains.txt" \
    "$S2D_RAW/mdcath/mdCATH_domains.txt"

# --- 6. ATLAS system list (MD ensemble source) — try known endpoints ---------
ATLAS_OK=0
for u in \
  "https://www.dsimb.inserm.fr/ATLAS/api/parsable" \
  "https://www.dsimb.inserm.fr/ATLAS/data/ATLAS/ATLAS.txt" \
  "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/ATLAS_info.csv" ; do
  if curl -fsL --max-time 120 -o "$S2D_RAW/atlas/atlas_list.part" "$u" 2>/dev/null \
     && [ -s "$S2D_RAW/atlas/atlas_list.part" ]; then
     mv "$S2D_RAW/atlas/atlas_list.part" "$S2D_RAW/atlas/atlas_list.raw"
     echo "[atlas] got $u"; log "OK" "$(stat -c%s "$S2D_RAW/atlas/atlas_list.raw")" "atlas_list.raw <- $u"
     ATLAS_OK=1; break
  fi
  rm -f "$S2D_RAW/atlas/atlas_list.part"
done
[ "$ATLAS_OK" = 0 ] && log "FAIL" "0" "atlas_list (all endpoints) — E0 will fall back to empirical+mdCATH"

echo "=========== E0 fetch done @ $(date -Is) ==========="
echo "manifest: $MANIFEST"; column -t -s$'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"
