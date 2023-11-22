#!/bin/bash
#SBATCH --job-name=bcl-aptos           # Job name
#SBATCH --output=slurm-logs/bcl-aptos.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=16          # Number of CPU cores
#SBATCH --gres=gpu:1               # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

sh train-aptos.sh
