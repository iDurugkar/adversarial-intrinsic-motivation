#!/bin/bash

#SBATCH --job-name opt_push				# Job name

### Logging
#SBATCH --output=/scratch/cluster/ishand/results/zoo2/push_opt_%A_%a.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/ishand/results/zoo2/push_opt_%A_%a.err                        # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job

### Node info
#SBATCH --partition titans                                                   # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                        # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 168:00:00                                                     # Run time (hh:mm:ss)

#SBATCH --gres=gpu:1                                                       # Number of gpus needed
#SBATCH --mem=10G                                                         # Memory requirements
#SBATCH --cpus-per-task=8                                              # Number of cpus needed per task

sleep $(($SLURM_ARRAY_TASK_ID))
python -u train.py --algo her --env FetchPush-v1 -n 500000 -optimize --n-trials 50 --n-jobs 2 --sampler tpe --pruner median --tensorboard-log /scratch/cluster/ishand/results/zoo2/aimher_rew_04/Push -f /scratch/cluster/ishand/results/zoo2/aimher_rew_04/Push &

wait

