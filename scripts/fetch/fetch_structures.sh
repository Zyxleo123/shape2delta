#!/bin/bash
# Fetch per-target sequences + AlphaFold models for the feature side (T0 seq/ESM,
# T1 static structure). LOGIN NODE. Idempotent. Targets come from the E0
# availability table (accession column). Small: ~700 FASTA + ~700 AF PDB + PAE.
set -uo pipefail
source "${S2D_ROOT:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh"

AVAIL="$S2D_RESULTS/E0_target_availability.csv"
[ -s "$AVAIL" ] || { echo "need $AVAIL (run E0 first)"; exit 1; }

# AFDB's file server (GCS-backed) returns 403 to curl without a User-Agent.
UA='Mozilla/5.0 (compatible; shape2delta-research)'
SEQDIR="$S2D_RAW/uniprot/fasta"; AFDIR="$S2D_RAW/alphafold/models"
mkdir -p "$SEQDIR" "$AFDIR"
SPX="$S2D_FEATURES/sciplex3/target_accessions.txt"   # sci-Plex3 single targets, if built
ACCS=$( { tail -n +2 "$AVAIL" | cut -d, -f1; [ -s "$SPX" ] && cat "$SPX"; } | sort -u)
n=$(echo "$ACCS" | wc -w); echo "[struct] $n target accessions"

got=0; miss_af=0
for acc in $ACCS; do
  # UniProt sequence (FASTA)
  fa="$SEQDIR/$acc.fasta"
  if [ ! -s "$fa" ]; then
    curl -fsL --max-time 60 -o "$fa.part" "https://rest.uniprot.org/uniprotkb/$acc.fasta" \
      && mv "$fa.part" "$fa" || rm -f "$fa.part"
  fi
  # AlphaFold model (PDB) + PAE, via the prediction API (gives exact versioned URLs)
  pdb="$AFDIR/$acc.pdb"
  if [ ! -s "$pdb" ]; then
    js=$(curl -fsL -A "$UA" --max-time 60 "https://alphafold.ebi.ac.uk/api/prediction/$acc" 2>/dev/null)
    purl=$(echo "$js" | grep -oE '"pdbUrl":"[^"]+"' | head -1 | sed 's/"pdbUrl":"//;s/"$//')
    eurl=$(echo "$js" | grep -oE '"paeDocUrl":"[^"]+"' | head -1 | sed 's/"paeDocUrl":"//;s/"$//')
    if [ -n "$purl" ]; then
      curl -fsL -A "$UA" --max-time 120 -o "$pdb.part" "$purl" && mv "$pdb.part" "$pdb" || rm -f "$pdb.part"
      [ -n "$eurl" ] && curl -fsL -A "$UA" --max-time 120 -o "$AFDIR/$acc.pae.json" "$eurl" 2>/dev/null || true
    else
      miss_af=$((miss_af+1))
    fi
  fi
  [ -s "$pdb" ] && got=$((got+1))
done
echo "[struct] AF models present: $got ; AF missing: $miss_af ; fasta: $(ls "$SEQDIR"/*.fasta 2>/dev/null | wc -l)"
