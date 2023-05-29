#!/bin/bash
#SBATCH --partition=scaling_data_pruning
#SBATCH --job-name=50openclip
#SBATCH --nodes 22
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8
#SBATCH --output=/checkpoints/amroabbas/datapruning/openclip/random-pruning/jobs/50/%x_%j.out
#SBATCH --error=/checkpoints/amroabbas/datapruning/openclip/random-pruning/jobs/50/%x_%j.error
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --time=1500

##SBATCH --exclude=a100-st-p4d24xlarge-477,a100-st-p4d24xlarge-820,a100-st-p4d24xlarge-707,a100-st-p4d24xlarge-879,a100-st-p4d24xlarge-426,a100-st-p4d24xlarge-437,a100-st-p4d24xlarge-442,a100-st-p4d24xlarge-793,a100-st-p4d24xlarge-444,a100-st-p4d24xlarge-835,a100-st-p4d24xlarge-431,a100-st-p4d24xlarge-579

export MASTER_PORT=12802

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export PYTHONPATH="$PYTHONPATH:$PWD/src"

EXP_NAME="openclip-b16-50%-rand-laion440m-22n-196bs_cont4" 

PRUNING_RATIO=0.5

TAG="neg_50"

RESUME="/checkpoint/amroabbas/datapruning/openclip/random-pruning/openclip-b16-50%-rand-laion440m-22n-196bs_cont3/neg_openclip-b16-50%-rand-laion440m-22n-196bs_cont3/checkpoints/epoch_28.pt"

srun --cpu_bind=v --accel-bind=gn python -u src/training/main.py \
    --subset-file "/data/home/amroabbas/projects/data_pruning/multimodal-repo/clustering/random_pruning_files/trie_random_${TAG}p.pickle" \
    --prune-ratio ${PRUNING_RATIO} \
    --save-frequency 1 \
    --report-to wandb \
    --train-data "/datasets01/laion2B-cvpr-filtered/shards/laion2B-en-joined{0..127}/{00000..00362}.tar" \
    --warmup 2000 \
    --batch-size 192 \
    --dataset-type webdataset \
    --epochs 32 \
    --workers 6 \
    --model ViT-B-16 \
    --seed 0 \
    --name ${EXP_NAME} \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --log-local \
    --logs "/checkpoints/amroabbas/datapruning/openclip/random-pruning/${EXP_NAME}" \
    --resume ${RESUME}