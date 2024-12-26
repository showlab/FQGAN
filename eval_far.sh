# !/bin/bash
set -x

torchrun \
--nnodes=1 \
--nproc_per_node=8 \
--node_rank=0 \
--master_port=12940 \
autoregressive/sample/sample_c2i_ddp_dual_code.py \
--vq-ckpt ./results_tokenizer_image/VQ-16-FQGAN_Dual_16384_40ep_DisAndSem/checkpoints/0200000.pt \
--codebook-size 16384 \
--codebook-embed-dim 8 \
--gpt-ckpt ./results/FAR-B-far_base_dual_code_300epoch/checkpoints/1500000.pt \
--gpt-model FAR-B \
--image-size 256 \
--image-size-eval 256 \
--cfg-scale 2.0 \
--top-k 0 \
--top-p 1.0 \
--sample-dir samples_far_base_dual_code_300epoch

python3 evaluations/evaluator.py \
reconstructions/VIRTUAL_imagenet256_labeled.npz \
samples_far_base_dual_code_300epoch/FAR-B-1500000-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz

