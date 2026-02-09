#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <input.qv> <output_dir> <num_shards> [proteinmpnn args...]" >&2
  exit 1
fi

INPUT_QV="$1"
OUTPUT_DIR="$2"
NUM_SHARDS="$3"
shift 3

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "$NUM_SHARDS" ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) must be < num_shards ($NUM_SHARDS)" >&2
  exit 1
fi

SHARD_DIR="$OUTPUT_DIR/input_shards"
mkdir -p "$SHARD_DIR"

SPLIT_DONE="$SHARD_DIR/.split.done"
SPLIT_LOCK="$SHARD_DIR/.split.lock"

if [ ! -f "$SPLIT_DONE" ]; then
  if mkdir "$SPLIT_LOCK" 2>/dev/null; then
    TOTAL_TAGS="$(uv run qvls "$INPUT_QV" | wc -l | tr -d ' ')"
    if [ "$TOTAL_TAGS" -eq 0 ]; then
      echo "Input quiver has no tags: $INPUT_QV" >&2
      exit 1
    fi
    NTAGS_PER_SHARD=$(( (TOTAL_TAGS + NUM_SHARDS - 1) / NUM_SHARDS ))
    uv run qvsplit "$INPUT_QV" "$NTAGS_PER_SHARD" --output-dir "$SHARD_DIR" --prefix split
    touch "$SPLIT_DONE"
    rmdir "$SPLIT_LOCK"
  else
    while [ ! -f "$SPLIT_DONE" ]; do
      sleep 2
    done
  fi
fi

INPUT_SHARD="$SHARD_DIR/split_${TASK_ID}.qv"
OUTPUT_SHARD="$OUTPUT_DIR/proteinmpnn_shard_${TASK_ID}.qv"

if [ ! -f "$INPUT_SHARD" ]; then
  echo "Missing shard input: $INPUT_SHARD" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Running ProteinMPNN shard $TASK_ID: $INPUT_SHARD -> $OUTPUT_SHARD"
uv run proteinmpnn \
  --input-quiver "$INPUT_SHARD" \
  --output-quiver "$OUTPUT_SHARD" \
  "$@"
