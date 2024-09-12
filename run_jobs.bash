#!/bin/bash

# Array of tasks
tasks=("go_to_goal" "push_box" "press_buttons")

# Loop over tasks and seeds
for task in "${tasks[@]}"; do
  for seed in {0..4}; do
    # Submit the job for each combination of task and seed
    sbatch <<EOT
#!/bin/bash
#SBATCH --account=ls_krausea
#SBATCH --job-name=Train_${task}_Seed${seed}      # Job name for each task/seed combination
#SBATCH --output=train_${task}_seed${seed}_%j.log # Output log for each job
#SBATCH --error=train_${task}_seed${seed}_%j.err  # Error log for each job
#SBATCH --gpus=rtx_2080:1
#SBATCH --cpus-per-task=20                         # Number of CPU cores
#SBATCH --mem-per-cpu=10240
#SBATCH --time=4:00:00                           # Time limit
#SBATCH --requeue                                 # Automatically requeue job if it fails

# Run the Python script for this task and seed
python3 SafeDreamer/train.py \
    --configs bsrp_lag \
    --method bsrp_lag \
    --jax.logical_gpus 0 \
    --task safeadaptationgym_point_${task} \
    --seed ${seed} \
    --run.steps 5e6 \
    --envs.amount 10 \
    --logdir /cluster/scratch/yardas/safedreamer/logdir_${task}_seed${seed}
EOT

  done
done
