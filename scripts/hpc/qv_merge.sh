#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <output.qv> <input_1.qv> [input_2.qv ...]" >&2
  exit 1
fi

OUTPUT_QV="$1"
shift

mkdir -p "$(dirname "$OUTPUT_QV")"
: > "$OUTPUT_QV"

for INPUT_QV in "$@"; do
  if [ ! -f "$INPUT_QV" ]; then
    echo "Missing input quiver: $INPUT_QV" >&2
    exit 1
  fi
  cat "$INPUT_QV" >> "$OUTPUT_QV"
  printf '\n' >> "$OUTPUT_QV"
done

echo "Merged $# shard file(s) into $OUTPUT_QV"
