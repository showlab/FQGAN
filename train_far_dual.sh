# !/bin/bash
set -x

# First, extract dual code from FQGAN
torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12335 \
autoregressive/train/extract_codes_c2i_dual_code.py \
--vq-ckpt ./results_tokenizer_image/VQ-16-FQGAN_Dual_16384_40ep_DisAndSem/checkpoints/0200000.pt \
--codebook-size 16384 \
--codebook-embed-dim 8 \
--data-path /home/ubuntu/DATA/ImageNet/train \
--code-path ./DATA/C2I/imagenet_code_c2i_flip_ten_crop_DualCode16384_40epoch \
--ten-crop \
--crop-range 1.1 \
--image-size 256


nnodes=1
nproc_per_node=8
node_rank=0
master_addr="localhost"
master_port=18988

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_c2i_dual.py \
--code-path /mnt/bn/vgfm2/test_dit/zechen/LlamaGen/DATA/C2I/imagenet_code_c2i_flip_ten_crop_DualCode16384_40epoch \
--image-size 256 \
--vocab-size 16384 \
--epochs 300 \
--dataset imagenet_dual_code \
--gpt-model FAR-B \
--ckpt-every 125000 \
--lr 1e-4 \
--global-batch-size 256 \
--exp-name far_base_dual_code_300epoch
