#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="$REPO_ROOT/slides-src"
PUBLIC_DIR="$REPO_ROOT/slides"
SOURCE_QMD="$SOURCE_DIR/on-manifold-tfg.qmd"
SOURCE_HTML="$SOURCE_DIR/on-manifold-tfg.html"
SOURCE_FILES="$SOURCE_DIR/on-manifold-tfg_files"

cd "$REPO_ROOT"

if rg -n '^[[:space:]]*:::+[[:space:]]*(\{[^}]*\.notes|notes)|data-notes:' "$SOURCE_QMD"; then
  echo "error: speaker notes must stay in .speaker-notes/, not the public QMD" >&2
  exit 1
fi

quarto render "$SOURCE_QMD"
python3 scripts/strip_speaker_notes.py "$SOURCE_HTML"

cp "$SOURCE_HTML" "$PUBLIC_DIR/index.html"
cp "$SOURCE_DIR/on-manifold-tfg.css" "$PUBLIC_DIR/on-manifold-tfg.css"
rsync -a --delete "$SOURCE_FILES/" "$PUBLIC_DIR/on-manifold-tfg_files/"

if rg -n '<aside class="notes"|data-notes=' "$PUBLIC_DIR/index.html"; then
  echo "error: speaker notes survived the public build" >&2
  exit 1
fi

python3 scripts/check_site_assets.py "$PUBLIC_DIR"
echo "Built note-free presentation: $PUBLIC_DIR/index.html"
