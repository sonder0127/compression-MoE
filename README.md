
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

> 以上摘要基于论文中的方法与实验小结整理（详见论文正文与图示）。

---

## 🧩 Methods

* **Low-rank Decomposition (D)**：对专家权重进行 SVD 截断得到低秩近似。
* **Expert Pruning (P)**：据多种专家重要性度量（激活频率、路由权重累计、输出 L2 范数等）剪除低贡献专家。
* **Expert Merging (M)**：先分组再融合，支持**主导专家聚类**与**全局相似聚类**；对应距离度量可选 **L2 / Cos**，融合规则 **LERP / SLERP**。
* **Sequential Compositions**：如 `D+P`、`M+D`、`P+M`，以及三法如 `P+M_L2+D`。
* **Contribution Allocation**：在固定总压缩率下，为各方法分配其贡献占比，形成二维/三维搜索面。

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
pip install -e .
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

## 📈 Reproducing Key Figures

> 把论文里对应图片导出为 PNG，放到 `assets/` 后用下方 Markdown 引用即可。

* **Pairwise vs. Single**
  ![Pairwise vs Single](assets/fig2_pairwise.png)
  *每列展示归一化的 perplexity 与 accuracy；阴影表示贡献区间极值，虚线为最优点。*

* **Best of Single/Pairwise/Triple**
  ![Single vs Pairwise vs Triple](assets/fig3_best_curves.png)

* **Order Sensitivity Heatmaps**
  ![Order Heatmaps](assets/fig4_order_heatmaps.png)

* **Decomposition ↔ Pruning 交互分析**
  ![Set Overlap after Decomposition vs Pruning](assets/fig5_overlap.png)

* **Speed & Peak Memory**
  ![Speed & Memory](assets/fig7_speed_memory.png)

* **Best Strategy by Compression Level**
  ![Best strategy grid](assets/fig8_best_grid.png)

---

## 🧪 Datasets & Tasks

* 语言建模：**WikiText-2**, **PTB**
* QA：**OpenBookQA**, **ARC-Easy**, **ARC-Challenge**, **MathQA**
* 评测指标：**Perplexity / Accuracy**；同时统计**吞吐 tokens/s**与**峰值显存**。
* 环境：示例在 **NVIDIA A100** 上完成，你也可以在其他 GPU 上复现（批大小与精度可能需调整）。

---

## 🧠 Practical Guidelines

* **Parsimony**：压缩率低时，单一方法常足够；压缩率升高再考虑组合。
* **Order**：`M→D` 往往优于 `D→M`；`D↔P` 顺序影响较小但推荐先 D 后 P；`M→P` 往往更差。
* **Balance**：在可压缩预算内，**均衡**的 D 与 P 往往更优；过度合并会损伤后续分解效果；只合并或只剪枝在激活专家数仍 ≥ top-k 时，对速度提升有限。

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
