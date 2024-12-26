nnodes=1
nproc_per_node=8
node_rank=0
master_addr="localhost"
master_port=8988

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
tokenizer/vq_train_dual.py \
  --data-path /home/ubuntu/DATA/ImageNet/train \
  --image-size 256 \
  --vq-model VQ-16 \
  --global-batch-size 256 \
  --codebook-size 16384 \
  --codebook-embed-dim 8 \
  --entropy-loss-ratio 0.03 \
  --epochs 40 \
  --lr 2e-4 \
  --with_disentanglement \
  --disentanglement-ratio 0.1 \
  --with_clip_supervision \
  --semantic-weight 0.5 \
  --exp-name FQGAN_Dual_16384_40ep_DisAndSem

