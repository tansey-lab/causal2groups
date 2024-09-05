#!/bin/bash
#SBATCH --job-name=add          
#SBATCH --partition=componc_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --array=0-99                                                                                              
#SBATCH --output=output_logs/add_%a.out  
#SBATCH --error=output_logs/add_%a.err  

source /home/toshc/miniconda3/bin/activate
conda activate c2g

python benchmarks/simulation.py --worker_id $SLURM_ARRAY_TASK_ID --n_workers 100 --setting "additive"
