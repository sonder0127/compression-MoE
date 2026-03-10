# Parsimony, Order and Balance: Principles for Compressing Mixture-of-Experts Models

Open-source implementation for compressing Mixture-of-Experts (MoE) models with **low-rank decomposition (D)**, **expert pruning (P)**, and **expert merging (M)**. The repository supports **single-method compression**, **sequential combinations**, and **contribution-ratio allocation** across methods.

- **Paper:** *Parsimony, Order, and Balance: Principles for Compressing Mixture-of-Experts Models*
- **Authors:** Shuangyou Feng, Chenyu Xu, Yi Ding, Zhi Liang, Sihai Zhang
- **Code:** [compression-MoE](https://github.com/sonder0127/compression-MoE)

## Overview

![Figure 1: Overview of compression strategies and sequential compositions](assets/Methodlogy.png)
*Figure 1. Overview of our MoE compression framework, including single methods, sequential combinations of D/P/M, and contribution-ratio allocation.*

![Figure 8: Best strategy by compression level and method mix](assets/Conclusion_heatmap.png)
*Figure 8. Best-performing strategies across compression levels. At higher compression ratios, more balanced contribution allocations often perform better.*

## TL;DR

- We systematically study **single methods**, **pairwise combinations**, and **triple sequential compositions** for MoE compression under different **target compression ratios** and **method-contribution allocations**.
- Based on extensive experiments, we summarize three practical principles for MoE compression: **Parsimony, Order, and Balance**.
- **Decomposition + Pruning (D + P)** is generally **synergistic** and works well in either order.
- **Merging + Decomposition (M + D)** is strongly **order-sensitive**, with **M → D** usually outperforming **D → M**.
- **Pruning + Merging (P + M)** is often **neutral or antagonistic**, and **M → P** is frequently inferior.
- At **low compression ratios**, single methods are often sufficient. At **higher compression ratios**, carefully designed combinations become more effective.
- In many high-compression settings, **balanced contribution allocations** produce the best overall trade-off.
- In terms of efficiency, higher compression reduces **peak memory usage**, while increasing the share of **decomposition** consistently improves **inference throughput**. When the number of remaining experts per layer is still no smaller than **top-k**, pruning-only or merging-only typically provides limited speedup.

## Project Structure

```text
compression-MoE/
├─ component/
│  ├─ modeling_compressdeepseek.py   # Compression wrappers for DeepSeek-MoE
│  ├─ modeling_compressqwen.py       # Compression wrappers for Qwen-MoE
│  ├─ modeling_compressolmoe.py      # Compression wrappers for OLMoE
│  ├─ modeling_merging.py            # Expert grouping and merging (dominant/global; LERP/SLERP)
│  ├─ modeling_pruning.py            # Expert pruning utilities (scoring and mask application)
│  └─ modeling_svdmlp.py             # SVD-based factorization for expert MLP weights
├─ configs/
│  └─ PMD.json                       # Hyperparameter presets for pruning/merging/SVD sweeps
├─ utils/
│  ├─ data_utils.py                  # Dataset loading utilities
│  └─ model_utils.py                 # Model loading utilities
├─ acc_evaluator.py                  # Accuracy evaluation for QA benchmarks
├─ calcute.py                        # Contribution-ratio calculator
├─ compress_method.py                # Compression method registry and orchestration
├─ ppl_evaluator.py                  # Perplexity evaluation for language modeling tasks
├─ run.py                            # Main CLI entry for the compression pipeline
└─ README.md
````

## Installation

```bash
# 1. Create the environment
conda env create -f openmax_base.yml
conda activate openmax

# 2. Install lm-evaluation-harness
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

# 3. Install additional dependencies
python -m pip install -r requirements.txt
```

## Quick Start

First, modify the configuration file according to your compression setting:

```bash
configs/PMD.json
```

Then run:

```bash
python run.py
```

## Supported Datasets and Tasks

| Task Type | Dataset |
|-----------|---------|
| Language Modeling | WikiText-2 |
| Language Modeling | PTB |
| Question Answering | OpenBookQA |
| Question Answering | ARC-Easy |
| Question Answering | ARC-Challenge |
| Question Answering | MathQA |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Perplexity | Language modeling performance |
| Accuracy | Downstream task performance |
| Throughput (tokens/s) | Inference speed |
| Peak memory usage | Memory efficiency during inference |

## Hardware Environment

Our experiments are primarily conducted on **NVIDIA A100 GPUs**. Other GPU platforms are also supported, although you may need to adjust the batch size, precision, or evaluation settings accordingly.

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@inproceedings{feng2025parsimony,
  title     = {Parsimony, Order and Balance: Principles for Compressing Mixture-of-Experts Models},
  author    = {Feng, Shuangyou and Xu, Chenyu and Ding, Yi and Liang, Zhi and Zhang, Sihai},
  booktitle = {Proceedings of ICASSP},
  year      = {2026},
  note      = {Code: \url{https://github.com/sonder0127/compression-MoE}}
}
```

## Acknowledgements

This project builds on **OLMoE** for the base MoE architecture and routing setup. The evaluation datasets are drawn from publicly available benchmarks.
