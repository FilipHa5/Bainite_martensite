#!/bin/bash -l
## Nazwa zlecenia
#SBATCH -J ADFtestjob
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=16GB
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=00:10:00 
## Nazwa grantu do rozliczenia zużycia zasobów CPU
#SBATCH -A plgstaleimetale-gpu-a100
## Specyfikacja partycji
#SBATCH --gres=gpu:1
#SBATCH -p plgrid-gpu-a100
## Plik ze standardowym wyjściem
#SBATCH --output="output.out"
## Plik ze standardowym wyjściem błędów
#SBATCH --error="error.err"

## przejscie do katalogu z ktorego wywolany zostal sbatch
cd $SLURM_SUBMIT_DIR
UV_CACHE_DIR=$SCRATCH uv run main.py

