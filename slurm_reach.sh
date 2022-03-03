#!/bin/bash

#SBATCH --job-name gail_reach				# Job name

### Logging
#SBATCH --output=/scratch/cluster/ishand/results/zoo2/gail_reach_%A_%a.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/ishand/results/zoo2/gail_reach_%A_%a.err                        # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job

### Node info
#SBATCH --partition titans                                                   # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                        # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 1:00:00                                                     # Run time (hh:mm:ss)

#SBATCH --gres=gpu:1                                                       # Number of gpus needed
#SBATCH --mem=12G                                                         # Memory requirements
#SBATCH --cpus-per-task=8                                              # Number of cpus needed per task

SEED=$(($SLURM_ARRAY_TASK_ID + 1010))
SEED2=$(($SEED + 3))
FILE=r_gail_td3

sleep $SLURM_ARRAY_TASK_ID * 3
python -u train.py --algo her --env FetchReach-v1 --tensorboard-log /scratch/cluster/ishand/results/zoo2/$FILE --eval-episodes 100 --eval-freq 2000 -f /scratch/cluster/ishand/results/zoo2/$FILE --seed $SEED &
sleep $SLURM_ARRAY_TASK_ID * 7
python -u train.py --algo her --env FetchReach-v1 --tensorboard-log /scratch/cluster/ishand/results/zoo2/$FILE --eval-episodes 100 --eval-freq 2000 -f /scratch/cluster/ishand/results/zoo2/$FILE --seed $SEED2 &

wait

