#!/bin/bash
#!
#! SLURM job script for TabICL fine-tuning on Wilkes3
#!

#SBATCH -J finetune-no-rope-short
#SBATCH -A MLMI-sv533-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mail-type=NONE
#SBATCH -p ampere

set -euo pipefail

# Basic module environment
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

# Python environment
PYTHON_EXEC="$HOME/.conda/envs/tabicl/bin/python"

# Work directory
REPO_DIR="/home/sv533/rds/hpc-work/tabicl"
cd "$REPO_DIR"

# Create needed folders
mkdir -p logs
mkdir -p "$REPO_DIR/wandb"
mkdir -p "$REPO_DIR/prior/mix_scm_no_rope_short"
mkdir -p "$REPO_DIR/checkpoints/finetune_no_rope_short"

# Logging
JOBID=$SLURM_JOB_ID
exec > "logs/out.$JOBID" 2>&1

echo "JobID: $JOBID"
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Python: $PYTHON_EXEC"
"$PYTHON_EXEC" --version

export OMP_NUM_THREADS=1

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------

# torchrun --standalone --nproc_per_node=1 /home/sv533/rds/hpc-work/tabicl/src/tabicl/train/run.py \
#             --wandb_log True \
#             --wandb_project TabICL \
#             --wandb_name Stage3 \
#             --wandb_dir /home/sv533/rds/hpc-work/tabicl/wandb \
#             --wandb_mode online \
#             --device cuda \
#             --dtype float32 \
#             --np_seed 42 \
#             --torch_seed 42 \
#             --max_steps 50 \
#             --batch_size 512 \
#             --micro_batch_size 1 \
#             --lr 1e-5 \
#             --scheduler constant \
#             --gradient_clipping 1.0 \
#             --prior_type mix_scm \
#             --prior_device cpu \
#             --batch_size_per_gp 1 \
#             --min_features 2 \
#             --max_features 100 \
#             --max_classes 10 \
#             --min_seq_len 40000 \
#             --max_seq_len 60000 \
#             --log_seq_len True \
#             --seq_len_per_gp True \
#             --replay_small True \
#             --min_train_size 0.5 \
#             --max_train_size 0.9 \
#             --embed_dim 128 \
#             --col_num_blocks 3 \
#             --col_nhead 4 \
#             --col_num_inds 128 \
#             --freeze_col True \
#             --row_num_blocks 3 \
#             --row_nhead 8 \
#             --row_num_cls 4 \
#             --row_rope_base 100000 \
#             --freeze_row False \
#             --icl_num_blocks 12 \
#             --icl_nhead 4 \
#             --ff_factor 2 \
#             --norm_first True \
#             --checkpoint_dir /home/sv533/rds/hpc-work/tabicl/checkpoints/finetune_base \
#             --checkpoint_path /home/sv533/rds/hpc-work/tabicl/checkpoints/tabicl-stage3.ckpt \
#             --save_temp_every 1 \
#             --save_perm_every 5 \
#             --only_load_model True


# ------------------------------------------------------
# Save prior datasets to disk and load them for training
# ------------------------------------------------------
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

# Saving to disk
"$PYTHON_EXEC" -m tabicl.prior.genload \
    --save_dir "$REPO_DIR/prior/mix_scm_no_rope_short" \
    --np_seed 42 \
    --torch_seed 42 \
    --num_batches 200 \
    --resume_from 0 \
    --batch_size 32 \
    --batch_size_per_gp 1 \
    --prior_type mix_scm \
    --min_features 2 \
    --max_features 100 \
    --max_classes 10 \
    --min_seq_len 512 \
    --max_seq_len 2048 \
    --log_seq_len True \
    --seq_len_per_gp True \
    --replay_small True \
    --min_train_size 0.5 \
    --max_train_size 0.9 \
    --n_jobs 1 \
    --num_threads_per_generate 1 \
    --device cpu

# Loading from disk and training
"$PYTHON_EXEC" -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m tabicl.train.run \
    --wandb_log False \
    --wandb_project TabICL \
    --wandb_name finetune_no_rope_short \
    --wandb_dir "$REPO_DIR/wandb" \
    --wandb_mode offline \
    --device cuda \
    --dtype float32 \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps 100 \
    --batch_size 32 \
    --micro_batch_size 1 \
    --lr 1e-5 \
    --scheduler constant \
    --gradient_clipping 1.0 \
    --prior_dir "$REPO_DIR/prior/mix_scm_no_rope_short" \
    --load_prior_start 0 \
    --delete_after_load False \
    --prior_device cpu \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --freeze_col True \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --freeze_row False \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --checkpoint_dir "$REPO_DIR/checkpoints/finetune_no_rope_short" \
    --checkpoint_path "$REPO_DIR/checkpoints/tabicl-stage3.ckpt" \
    --save_temp_every 25 \
    --save_perm_every 50 \
    --only_load_model True