<div align="center">
<br>
<h3>Factorized Visual Tokenization and Generation</h3>

[Zechen Bai](https://www.baizechen.site/) <sup>1</sup>&nbsp;
[Jianxiong Gao](https://jianxgao.github.io/) <sup>2</sup>&nbsp;
[Ziteng Gao](https://sebgao.github.io/) <sup>1</sup>&nbsp;
[Pichao Wang](https://wangpichao.github.io/) <sup>3</sup>&nbsp;
[Zheng Zhang](https://scholar.google.com/citations?user=k0KiE4wAAAAJ&hl=en) <sup>3</sup>&nbsp;
[Tong He](https://hetong007.github.io/) <sup>3</sup>&nbsp;
[Mike Zheng Shou](https://sites.google.com/view/showlab) <sup>1</sup>&nbsp;

arXiv 2024

<sup>1</sup> [Show Lab, National University of Singapore](https://sites.google.com/view/showlab/home) &nbsp; <sup>2</sup> Fudan University&nbsp; <sup>3</sup> Amazon&nbsp;
 
[![arXiv](https://img.shields.io/badge/arXiv-<2409.19603>-<COLOR>.svg)](https://arxiv.org/abs/2411.16681)

</div>

**News**
* **[2024-11-28]** The code and model will be released soon after internal approval!
* **[2024-11-26]** We released our paper on [arXiv](https://arxiv.org/abs/2411.16681).

## TL;DR
FQGAN is state-of-the-art visual tokenizer with a novel factorized tokenization design, surpassing VQ and LFQ methods in discrete image reconstruction.

<p align="center"> <img src="assets/rfid_teaser.jpg" width="555"></p>

## Method Overview

FQGAN addresses the large codebook usage issue by decomposing a single large codebook into multiple independent sub-codebooks.
By leveraging disentanglement regularization and representation learning objectives, the sub-codebooks learn hierarchical, structured and semantic meaningful representations.
FQGAN achieves state-of-the-art performance on discrete image reconstruction, surpassing VQ and LFQ methods.

<p align="center"> <img src="assets/framework.jpg" width="888"></p>


## Comparison with previous visual tokenizers
<p align="center"> <img src="assets/Tab_Tok.png" width="666"></p>

## What has each sub-codebook learned?
<p align="center"> <img src="assets/tsne_dual_codebook.jpg" width="666"></p>

<p align="center"> <img src="assets/recon_codebook.jpg" width="666"></p>

## Can this tokenizer be used into downstream image generation?

<p align="center"> <img src="assets/Tab_AR.png" width="666"></p>
<p align="center"> <img src="assets/AR_gen.jpg" width="888"></p>

## Citation
To cite the paper and model, please use the below:
```
@article{bai2024factorized,
  title={Factorized Visual Tokenization and Generation},
  author={Bai, Zechen and Gao, Jianxiong and Gao, Ziteng and Wang, Pichao and Zhang, Zheng and He, Tong and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2411.16681},
  year={2024}
}
```
