#!/bin/bash
#SBATCH --job-name=nocov          
#SBATCH --partition=componc_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --array=0-99                                                                                              
#SBATCH --output=output_logs/nocov_%a.out  
#SBATCH --error=output_logs/nocov_%a.err  

source /home/toshc/miniconda3/bin/activate
conda activate c2g

python src/causal2groups/no_covariates.py --index $SLURM_ARRAY_TASK_ID --n_jobs 100 --n_seeds 30
