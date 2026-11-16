#!/bin/bash
# Download the LINCS 2020 beta level-5 trt_cp signature matrix (33GB HDF5/GCTX).
# LOGIN NODE only. Resumable (curl -C -). Idempotent: verifies final size.
set -uo pipefail
source "${S2D_ROOT:-/zfsauton2/home/yixiz/shape2delta}/slurm/preamble.sh"

URL="https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx"
DEST="$S2D_RAW/lincs/level5_beta_trt_cp_n720216x12328.gctx"
EXPECT=35518405386

mkdir -p "$(dirname "$DEST")"
if [ -s "$DEST" ] && [ "$(stat -c%s "$DEST")" -eq "$EXPECT" ]; then
  echo "[lincs-l5] already complete: $DEST ($(stat -c%s "$DEST") bytes)"; exit 0
fi
echo "[lincs-l5] downloading $URL -> $DEST ($(date -Is))"
curl -fL -C - --retry 5 --retry-delay 10 --max-time 7200 -o "$DEST" "$URL"
got=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
echo "[lincs-l5] got=$got expect=$EXPECT"
if [ "$got" -ne "$EXPECT" ]; then echo "[lincs-l5] SIZE MISMATCH"; exit 1; fi
echo "[lincs-l5] complete @ $(date -Is)"
