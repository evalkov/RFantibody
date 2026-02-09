# HPC Parallelization Helpers

These scripts shard Quiver workloads for SLURM array jobs and merge shard outputs.

## Typical Pattern

1. Submit an array job for one stage (`proteinmpnn_array.sh` or `rf2_array.sh`).
2. Each task processes one shard (`$SLURM_ARRAY_TASK_ID`).
3. Merge shard outputs with `qv_merge.sh`.

## Scripts

- `rfdiffusion_array.sh`
  Runs RFdiffusion in shard mode by splitting `num_designs` across array tasks.
- `proteinmpnn_array.sh`
  Splits one input Quiver into N shard Quivers and runs ProteinMPNN on one shard.
- `rf2_array.sh`
  Splits one input Quiver into N shard Quivers and runs RF2 on one shard.
- `qv_merge.sh`
  Concatenates shard Quiver files into a single output Quiver.

## Example

```bash
# Stage 1: RFdiffusion with 8 array tasks
sbatch --array=0-7 scripts/hpc/rfdiffusion_array.sh \
  scripts/examples/example_inputs/flu_HA.pdb \
  scripts/examples/example_inputs/h-NbBCII10.pdb \
  /scratch/$USER/rfab/stage1 1000 8 \
  --design-loops H1:7,H2:6,H3:5-13 --hotspots B146,B170,B177 --no-trajectory

# Stage 2: ProteinMPNN with 8 array tasks
sbatch --array=0-7 scripts/hpc/proteinmpnn_array.sh \
  /scratch/$USER/rfab/stage1/merged.qv /scratch/$USER/rfab/stage2 8 \
  --seqs-per-struct 4 --temperature 0.2 --batch-size 4

# Stage 3: RF2 with 8 array tasks
sbatch --array=0-7 scripts/hpc/rf2_array.sh \
  /scratch/$USER/rfab/stage2/merged.qv /scratch/$USER/rfab/stage3 8 \
  --num-recycles 10 --hotspot-show-prop 0.0
```
