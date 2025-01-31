#!/bin/bash
#SBATCH --job-name=base          
#SBATCH --partition=componc_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=output_logs/base.out  
#SBATCH --error=output_logs/base.err  

source /home/toshc/miniconda3/bin/activate
conda activate c2g

python benchmarks/baseline_ite.py
