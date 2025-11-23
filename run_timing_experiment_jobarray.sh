#!/bin/bash
#SBATCH --job-name=GNO_timings
#SBATCH --output=./logs/GNO_%A_%a.log
#SBATCH --error=./logs/GNO_%A_%a.err
#SBATCH --partition=windq,windfatq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=infinite
#SBATCH --array=1-12



START_TIME=`date +%s`
NODE_ID=$(scontrol show hostnames $SLURM_JOB_NODELIST)

export LC_ALL=en_US.UTF-8


echo ------------------------------------------------------
echo Date: $(date)
echo Sophia job is running on node: ${NODE_ID}
echo Sophia job identifier: $SLURM_JOBID
echo ------------------------------------------------------ 


source ~/.bashrc
micromamba activate ml_GPU
python3 ./Experiments/articles_plotting/memory_consumption_calculation.py --timing_experiment_id $SLURM_ARRAY_TASK_ID

END_TIME=`date +%s`
echo ------------------------------------------------------
echo Finished job
echo "Elapsed time: $(($END_TIME-$START_TIME)) seconds"
echo ------------------------------------------------------
