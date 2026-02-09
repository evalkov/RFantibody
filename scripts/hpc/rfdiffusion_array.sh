#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <target.pdb> <framework.pdb> <output_dir> <total_designs> <num_shards> [rfdiffusion args...]" >&2
  exit 1
fi

TARGET_PDB="$1"
FRAMEWORK_PDB="$2"
OUTPUT_DIR="$3"
TOTAL_DESIGNS="$4"
NUM_SHARDS="$5"
shift 5

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -ge "$NUM_SHARDS" ]; then
  echo "SLURM_ARRAY_TASK_ID ($TASK_ID) must be < num_shards ($NUM_SHARDS)" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

DESIGNS_PER_SHARD=$(( (TOTAL_DESIGNS + NUM_SHARDS - 1) / NUM_SHARDS ))
START_NUM=$(( TASK_ID * DESIGNS_PER_SHARD ))
REMAINING=$(( TOTAL_DESIGNS - START_NUM ))

if [ "$REMAINING" -le 0 ]; then
  echo "No work for task $TASK_ID"
  exit 0
fi

COUNT="$DESIGNS_PER_SHARD"
if [ "$REMAINING" -lt "$COUNT" ]; then
  COUNT="$REMAINING"
fi

OUTPUT_QV="$OUTPUT_DIR/rfdiffusion_shard_${TASK_ID}.qv"

echo "Running shard $TASK_ID: start=$START_NUM count=$COUNT output=$OUTPUT_QV"
uv run rfdiffusion \
  --target "$TARGET_PDB" \
  --framework "$FRAMEWORK_PDB" \
  --output-quiver "$OUTPUT_QV" \
  --num-designs "$COUNT" \
  --extra "inference.design_startnum=$START_NUM" \
  "$@"
