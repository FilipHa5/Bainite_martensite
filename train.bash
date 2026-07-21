#!/bin/bash -l
#SBATCH -J MicrostructureNet
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=16GB
#SBATCH --time=48:00:00 
#SBATCH -A plgstaleimetale-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH -p plgrid-gpu-a100
#SBATCH --array=0-3
#SBATCH --output=output_%A_%a.out
#SBATCH --error=error_%A_%a.err


module load CUDA/11.7.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training on: $DATA_FILE → $DEST_DIR"

cd $SLURM_SUBMIT_DIR
UV_CACHE_DIR=$SCRATCH uv run main.py  --results=results_${SLURM_JOB_ID} --outer_fold $SLURM_ARRAY_TASK_ID