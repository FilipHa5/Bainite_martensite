#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J MicrostructureNet
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=4
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=16GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:18:00 
## Nazwa grantu do rozliczenia zużycia zasobów CPU
#SBATCH -A plgstaleimetale-gpu-a100
## Specyfikacja partycji
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
# nvidia-smi

cd $SLURM_SUBMIT_DIR
UV_CACHE_DIR=$SCRATCH uv run main.py  --results=results_${SLURM_JOB_ID} --outer_fold $SLURM_ARRAY_TASK_ID