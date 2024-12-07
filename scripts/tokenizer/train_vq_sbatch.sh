#!/bin/bash
#SBATCH --job-name=train_Llama
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=3-00:00:00

source ~/diff-torch/bin/activate
export nnodes=1
export nproc_per_node=4
export node_rank=0
export master_addr=$(hostname -I | awk '{print $1}')
export master_port=12345
export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@" 
