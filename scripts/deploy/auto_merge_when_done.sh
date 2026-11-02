#!/bin/bash
# Auto-merge completed ingest files into sft_master.jsonl
# Run once manually after all background ingests complete

cd /Users/dadsmacpro/Desktop/anthos-repo

FILES=(
  "data/art_media_sft.jsonl"
  "data/flagship_sft.jsonl"
  "data/stem_finance_corporate_sft.jsonl"
  "data/alignment_science_sft.jsonl"
  "data/music_cinema_sft.jsonl"  # already merged but idempotent check
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ] && [ -s "$f" ]; then
    lines=$(wc -l < "$f")
    echo "Merging $f ($lines pairs)..."
    cat data/sft_master.jsonl "$f" > /tmp/m.jsonl && mv /tmp/m.jsonl data/sft_master.jsonl
    echo "  Done. Master now: $(wc -l < data/sft_master.jsonl) pairs"
  else
    echo "Skipping $f (missing or empty)"
  fi
done

echo ""
echo "FINAL: $(wc -l < data/sft_master.jsonl) pairs | $(ls -lh data/sft_master.jsonl | awk '{print $5}')"
