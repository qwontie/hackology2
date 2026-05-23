#!/usr/bin/env bash
# ABOUTME: Download hackathon dataset from external hosting with checksum verification.
# ABOUTME: URL is a placeholder — replace DATA_URL before distributing to teams.
set -euo pipefail

# ============================================================================
# IMPORTANT: Replace this URL with the actual dataset download link before
# distributing the template to teams.
# ============================================================================
DATA_URL="${DATA_URL:-https://files.assecobs.pl/s/XHyAKRmBbrsiEZD/download}"

DATA_DIR="data"
ZIP_FILE="${DATA_DIR}/dataset.zip"
CHECKSUM_FILE="checksums.sha256"

echo "==> Creating data directory..."
mkdir -p "$DATA_DIR"

echo "==> Downloading dataset..."
curl -L --fail --progress-bar -o "$ZIP_FILE" "$DATA_URL"

echo "==> Verifying checksums..."
if [ -f "$CHECKSUM_FILE" ]; then
    sha256sum -c "$CHECKSUM_FILE"
else
    echo "WARNING: No checksums.sha256 file found — skipping verification."
    echo "         Generate checksums with: sha256sum ${ZIP_FILE} > ${CHECKSUM_FILE}"
fi

echo "==> Extracting dataset..."
unzip -o -q "$ZIP_FILE" -d "."

echo "==> Copying test_images.json to data directory..."
if [ -f "test_images.json" ]; then
    cp test_images.json "$DATA_DIR/test_images.json"
fi

echo "==> Cleanup..."
rm -f "$ZIP_FILE"

echo "==> Done. Dataset available in ${DATA_DIR}/"
ls -lh "$DATA_DIR"/
