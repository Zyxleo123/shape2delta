#!/bin/bash
# sci-Plex3 (Srivatsan et al. 2020, GSE139944) as the scPerturb-harmonised h5ad
# (Zenodo record 10044268 — same record as the Replogle file). LOGIN NODE.
# Idempotent: skips if the file is present with the expected size.
set -uo pipefail
source "${S2D_ROOT:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh"
OUT="$S2D_RAW/sciplex3"; mkdir -p "$OUT"
F="SrivatsanTrapnell2020_sciplex3.h5ad"
URL="https://zenodo.org/api/records/10044268/files/${F}/content"
WANT=$(curl -sL https://zenodo.org/api/records/10044268 \
       | python -c "import json,sys;print([f['size'] for f in json.load(sys.stdin)['files'] if f['key']=='$F'][0])")
HAVE=$(stat -c %s "$OUT/$F" 2>/dev/null || echo 0)
if [ "$HAVE" = "$WANT" ]; then echo "[sciplex3] present ($HAVE bytes)"; exit 0; fi
echo "[sciplex3] downloading $F ($WANT bytes) @ $(date -Is)"
curl -fL --retry 5 -C - -o "$OUT/$F.part" "$URL" && mv "$OUT/$F.part" "$OUT/$F"
echo "[sciplex3] done: $(stat -c %s "$OUT/$F") / $WANT bytes @ $(date -Is)"
