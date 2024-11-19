#!/bin/bash
#SBATCH --job-name=nonadd          
#SBATCH --partition=componc_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --array=0-99                                                                                            
#SBATCH --output=output_logs/nonadd_%a.out  
#SBATCH --error=output_logs/nonadd_%a.err  

source /home/toshc/miniconda3/bin/activate
conda activate c2g

python benchmarks/simulation.py --worker_id $SLURM_ARRAY_TASK_ID --n_workers 100 --setting "nonadditive"
