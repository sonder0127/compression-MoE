
---

# PARSIMONY, ORDER & BALANCE — Compression Principles for MoE

> Open-source implementation for compressing Mixture-of-Experts (MoE) models via **low-rank decomposition (D)**, **expert pruning (P)**, and **expert merging (M)**, with support for **sequential combinations** and **contribution-ratio allocation**.

* **Paper:** PARSIMONY, ORDER AND BALANCE: PRINCIPLES FOR COMPRESSING MIXTURE-OF-EXPERTS MODELS
* **Authors:** Shuangyou Feng, Chenyu Xu, Yi Ding, Zhi Liang, Sihai Zhang
* **Code:** [https://github.com/sonder0127/compression-MoE](https://github.com/sonder0127/compression-MoE)

<!-- ===== Figures at the beginning ===== -->
![Figure 1 — Overview of compression strategies and sequential compositions](assets/Methodlogy.png)
*Figure 1. Overview of our MoE compression setting and sequential combinations (D/P/M) with contribution allocations.*

![Figure 8 — Best strategy by compression level and method mix](assets/Conclusion_heatmap.png)
*Figure 8. Best-performing strategy across compression levels. Balanced contributions tend to win under higher compression.*

## 🔍 TL;DR

* 我们系统化评测了 MoE 压缩的单法、两两组合与三法顺序组合，并在不同**压缩率**与**方法贡献分配**下给出对比与原则。结论高度概括为：**Parsimony（节制）、Order（顺序）、Balance（均衡）**。
* **D 与 P**在任意顺序下**协同**；**M 与 D**对**顺序敏感**，通常 **M→D 优于 D→M**；**P 与 M**多为**中性或拮抗**，且 **M→P 往往更差**。
* 低压缩率下单法已足够；高压缩率下合理的组合更有效；**均衡的贡献分配**通常带来更优结果。
* 速度与显存：更高压缩降低峰值显存；提高**分解贡献**能稳定提升推理速度；当剩余专家数仍 ≥ top-k 时，纯剪枝或合并对速度提升有限。

---

## 🗂️ Project Structure

```
compression-MoE/
├─ component/
│  ├─ modeling_compressdeepseek.py   # DeepSeek-MoE compression wrappers
│  ├─ modeling_compressqwen.py       # Qwen-MoE compression wrappers
│  ├─ modeling_compressolmoe.py      # OLMoE compression wrappers
│  ├─ modeling_merging.py            # Expert grouping & merging (dominant/global; LERP/SLERP)
│  ├─ modeling_pruning.py            # Expert pruning utilities (scoring & mask application)
│  └─ modeling_svdmlp.py             # SVD-based factorization for expert MLP weights
├─ configs/
│  ├─ pruning_merging_svd.json       # Hyperparameter presets for pruning/merging/SVD sweeps
│  └─ ...                            # Additional experiment configs
├─ method/
│  ├─ acc_evaluator.py               # Accuracy evaluator for QA benchmarks
│  ├─ calcute.py                     # Contribution-ratio calculator (allocation utilities)
│  ├─ compress_method.py             # Compression method registry & orchestration
│  ├─ expert_pruning.py              # Implementations of expert-pruning strategies/metrics
│  └─ ppl_evaluator.py               # Perplexity evaluator for language-modeling tasks
├─ utils/
│  ├─ data_utils.py                  # Dataset loading
│  └─ model_utils.py                 # Model loading
├─ assets/                           # Figures and static assets for README/docs
├─ README.md
└─ pruning_merging_svd.py            # Main CLI entry for joint pruning/merging/SVD pipeline


```

---

## 📦 Installation

```bash
# 1) Create env
conda create -n moe-compress python=3.10 -y
conda activate moe-compress

# 2) 本仓库作为包
pip install -r requirements.txt
```

---

## 🚀 Quick Start
根据压缩需求修改配置文件

`configs/pruning_merging_svd.json`（示例）：

```json
{
    "model_name": "OLMoE",
    "model_path": "outfile/OLMoE/model_saved/ASVD_Seed=3.pt",
    "dataset": "wikitext2",
    "seed": 3,
    "DEV": "cpu",
    "step": 0,
    "updating_nsamples": 16,
  
    "eval": {
      "enabled": true,
      "batch_size": 512,
      "nsamples": 500,
      "wiki_batch_size": 16,
      "wiki_nsamples": 0,
      "gen_seq_len": 1024,
      "model_seq_len": 2048
    },
    "svd": {
      "enabled": true,
      "method": "ASVD",
      "mlp_rank": 300,
      "attn_rank": 2048,
      "compress_ratio": 0.1,
      "whitening_nsamples": 32,
      "load_from_file": false,
      "save_model": false
    },
    "pruning": {
      "enabled": true,
      "eval_nsamples": 256,
      "importance_metrics": "activation_frequency",
      "load_from_file": true,
      "strategy": "fixed",
      "compress_ratio": 0.1,
      "pruning_nsamples": 256,
      "save_model": false
    },
    "merging": {
      "enabled": true,
      "eval_nsamples": 32,
      "eval_object": "weight",
      "metrics": "l2",
      "weighting_factor": "activation_frequency",
      "load_from_file": true,
      "save_model": false,
      "num_expert_group": 59,
      "compress_ratio": 0.1
    },
    "other": {
      "build_basenet": false,
      "num_dominate_expert": 8,
      "fine_tune_path": null
    }
  }
```

---

然后run
```bash
python pruning_merging_svd.py
```

---

## 🧪 Datasets & Tasks

* 语言建模：**WikiText-2**, **PTB**
* QA：**OpenBookQA**, **ARC-Easy**, **ARC-Challenge**, **MathQA**
* 评测指标：**Perplexity / Accuracy**；**吞吐 tokens/s**与**峰值显存**。
* 环境：示例在 **NVIDIA A100** 上完成，你也可以在其他 GPU 上复现（批大小与精度可能需调整）。

---

## 📜 Citation

如果你觉得这个仓库对你有帮助，请引用我们的论文：

```bibtex
@inproceedings{feng2025parsimony,
  title     = {Parsimony, Order and Balance: Principles for Compressing Mixture-of-Experts Models},
  author    = {Feng, Shuangyou and Xu, Chenyu and Ding, Yi and Liang, Zhi and Zhang, Sihai},
  booktitle = {Proceedings of ICASSP},
  year      = {2026},
  note      = {Code: \url{https://github.com/sonder0127/compression-MoE}}
}
```

## 🙏 Acknowledgements

* 基座与路由设置参考 OLMoE；数据与任务来自公开数据集。
