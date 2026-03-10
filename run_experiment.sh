#!/bin/bash
#SBATCH --job-name=GNO-master
#SBATCH --output=./logs/GNO_%A.log
#SBATCH --error=./logs/GNO_%A.err
#SBATCH --partition=windq
#SBATCH --nodes=1
#SBATCH --time=INFINITE

# Master job runs on windq (infinite time) and distributes GPU work to fatq via Hydra submitit

START_TIME=$(date +%s)
NODE_ID=$(scontrol show hostnames $SLURM_JOB_NODELIST)

export LC_ALL=en_US.UTF-8

echo ------------------------------------------------------
echo Date: $(date)
echo Sophia master job is running on node: ${NODE_ID}
echo Sophia job identifier: $SLURM_JOBID
echo ------------------------------------------------------

# Setup pixi environment
export PIXI_PROJECT_ROOT=/work/users/jpsch/gno
eval "$(pixi shell-hook -e cluster)"

# Run experiment (pass config name as argument, defaults to test_GNO_probe)
CONFIG_NAME="${1:-test_GNO_probe}"
echo "Running config: ${CONFIG_NAME}"
echo "GPU jobs will be submitted to fatq via Hydra submitit"

python main.py --config-name "${CONFIG_NAME}" "${@:2}"

END_TIME=$(date +%s)
echo ------------------------------------------------------
echo Finished job
echo "Elapsed time: $(($END_TIME-$START_TIME)) seconds"
echo ------------------------------------------------------
