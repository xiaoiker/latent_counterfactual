# !/bin/bash
set -x
export nnodes=1
export nproc_per_node=1
export node_rank=0
export master_addr=$(hostname -I | awk '{print $1}')
export master_port=12345
export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/tokenizer_image/vq_train.py "$@"
