#coding:utf8
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
from component.modeling_compressdeepseek import SVD_DeepseekMLP
from component.modeling_merging import *
from component.modeling_compressolmoe import OlmoeImportanceWrapper,SVD_OlmoeMLP
from component.modeling_compressqwen import QwenMoeImportanceWrapper, SVD_Qwen2MoeMLP
from utils.model_utils import find_layers, prepare_inputs_for_model, prepare_inputs_for_qwen
from utils.data_utils import get_calib_train_data

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 模型相关
from transformers import (
    AutoTokenizer,
    Qwen2MoeForCausalLM
)
import numpy as np

#=============SVD================
@torch.no_grad()
def profile_svdllm(model, calib_loader, dev):
    layers = model.model.layers

    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat

def contains_inf_or_nan(tensor):
    return torch.any(torch.isnan(tensor)) or torch.any(torch.isinf(tensor))

@torch.no_grad()
def profile_asvd(model, calib_loader, dev, alpha=0.5):
    layers = model.model.layers
    model = model.to(dev)
    print("Start obtaining the scaling matrix S...")

    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:  # For opt models
            inp = inp.unsqueeze(0)
        
        # Calculate the mean absolute value of activations for each channel
        abs_mean_activations = torch.mean(torch.abs(inp), dim=(0, 1))
        
        # Accumulate the result in the module's raw_scaling_diag_matrix
        module.raw_scaling_diag_matrix += abs_mean_activations
        
        del inp
        torch.cuda.empty_cache()

    # Initialize raw_scaling_diag_matrix for all linear modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = torch.zeros(module.weight.size(1)).to(dev)
            module.register_forward_hook(hook)

    # Run the calibration dataset through the model to accumulate statistics
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)

    # Clear hooks and free memory
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()

    # Compute the scaling matrix S for each layer
    profiling_mat = {}
    print("Start computing the scaling matrix S...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            abs_mean_activations = subset[name].raw_scaling_diag_matrix.cpu()

            # Ensure no non-finite values in abs_mean_activations
            if contains_inf_or_nan(abs_mean_activations):
                print(f"Layer {i}, Module {name}: abs_mean_activations 包含 NaN 或 inf 值")
                abs_mean_activations[torch.isnan(abs_mean_activations)] = 0
                abs_mean_activations[torch.isinf(abs_mean_activations)] = 0

            # Compute Sii based on the average magnitude of activation in each channel
            Sii = (abs_mean_activations ** alpha)
            
            # Ensure no non-finite values in Sii
            if contains_inf_or_nan(Sii):
                print(f"Layer {i}, Module {name}: Sii 包含 NaN 或 inf 值")
                Sii[torch.isnan(Sii)] = 1e-6  # Small positive value
                Sii[torch.isinf(Sii)] = 1e-6  # Small positive value

            scaling_diag_matrix = torch.diag(Sii)

            # Ensure no non-finite values in scaling_diag_matrix
            if contains_inf_or_nan(scaling_diag_matrix):
                print(f"Layer {i}, Module {name}: scaling_diag_matrix 包含 NaN 或 inf 值")
                scaling_diag_matrix[torch.isnan(scaling_diag_matrix)] = 1e-6
                scaling_diag_matrix[torch.isinf(scaling_diag_matrix)] = 1e-6

            layer_profile[name] = scaling_diag_matrix
            del abs_mean_activations, Sii, scaling_diag_matrix
            torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile

    return profiling_mat



@torch.no_grad()
def multi_svd_implementaions(W, svd_method, layer_idx, lora_rank, name, profiling_mat, dev, expert_idx=0):
    dtype = W.dtype
    if svd_method == 'vanilla-SVD':
        U, S, VT = torch.linalg.svd(W, full_matrices=False)       # 对权重矩阵进行SVD分解
        # num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))    # 计算截断后的奇异值数量
        num_s_after_trunc = lora_rank
        truc_s = S[:num_s_after_trunc]  # 截断奇异值
        truc_u = U[:, :num_s_after_trunc]   # 截断U矩阵
        truc_v = VT[:num_s_after_trunc, :]    # 计算截断后的V矩阵
        truc_sigma = torch.diag(truc_s)     # 构建对角矩阵
        
        sqrtSigma = torch.sqrt(truc_sigma)      # 计算对角矩阵的平方根
        svd_u = torch.matmul(truc_u, sqrtSigma).to(dtype)     # 计算低秩分解后的U矩阵
        svd_v = torch.matmul(sqrtSigma, truc_v).to(dtype)     # 计算低秩分解后的V矩阵

    elif svd_method in ('SVD-LLM', 'ASVD'):
        if name in ('gate_proj', 'up_proj', 'down_proj'):
            key_name = f'mlp.experts.{expert_idx}.{name}'
        else:
            key_name = f'self_attn.{name}'

        scaling_diag_matrix = profiling_mat[layer_idx][key_name].to(dev)     # 获取当前层的缩放矩阵
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)  # 计算缩放矩阵的逆矩阵
        except Exception as e:
            #print("Warning: scaling_diag_matrix is not full rank!")     # 如果缩放矩阵不是满秩的，打印警告
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev) # 添加一个小的对角矩阵使其变为满秩
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)      # 再次计算逆矩阵
        scaling_diag_matrix = scaling_diag_matrix.float()   # 将缩放矩阵转换为浮点数类型
        scaling_matrix_inv = scaling_matrix_inv.float()     # 将逆矩阵转换为浮点数类型
        W_scale = torch.matmul(W, scaling_diag_matrix)      # 计算缩放后的权重矩阵


        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)       # 对权重矩阵进行SVD分解
        # num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))    # 计算截断后的奇异值数量
        num_s_after_trunc = lora_rank
        truc_s = S[:num_s_after_trunc]  # 截断奇异值
        truc_u = U[:, :num_s_after_trunc]   # 截断U矩阵
        truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)    # 计算截断后的V矩阵
        truc_sigma = torch.diag(truc_s)     # 构建对角矩阵
        
        sqrtSigma = torch.sqrt(truc_sigma)      # 计算对角矩阵的平方根
        svd_u = torch.matmul(truc_u, sqrtSigma).to(dtype)     # 计算低秩分解后的U矩阵
        svd_v = torch.matmul(sqrtSigma, truc_v).to(dtype)     # 计算低秩分解后的V矩阵
    
    W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
    del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma

    return svd_u, svd_v

@torch.no_grad()
def model_expert_only_SVD(model_name, model, svd_method, mlp_lora_rank, dev, profiling_mat=None, ratio=1):
            # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    layers = model.model.layers
    print("Start SVD decomposition...")
    for i in range(len(layers)):
        layer = layers[i]
        #### Replace MLP ####
        if "OLMoE" in model_name:
            svd_experts = nn.ModuleList([SVD_OlmoeMLP(config=model.config, ratio=ratio, lora_rank=mlp_lora_rank) for _ in range(layer.mlp.num_experts)])

        elif "Qwen" in model_name:
            # 未完成
            svd_experts = nn.ModuleList(
            [SVD_Qwen2MoeMLP(config=model.config, intermediate_size=model.config.moe_intermediate_size, ratio=ratio, lora_rank=mlp_lora_rank).to(dev) for _ in range(layer.mlp.num_experts)]
        )
        elif 'DeepSeek' in model_name:
            # 未完成
            svd_experts = nn.ModuleList(
            [SVD_DeepseekMLP(config=model.config, intermediate_size=model.config.moe_intermediate_size, ratio=ratio, lora_rank=mlp_lora_rank).to(dev) for _ in range(model.config.n_routed_experts)]
            )

        #### Replace MLP ####
        for n in range(len(layer.mlp.experts)):
            subset = find_layers(layer.mlp.experts[n])
            for name in subset:
                W = subset[name].weight.data.float().to(dev)    # 从子层中提取权重矩阵W并转换为浮点类型
                svd_u, svd_v = multi_svd_implementaions(W, svd_method, i, mlp_lora_rank, name, profiling_mat, dev, n)

                #### Replace MLP ####
                if model_name in ('OLMoE', 'Qwen', 'DeepSeek'):    
                    if "gate_proj" in name:
                        svd_experts[n].gate_u_proj.weight.data = svd_u.to(dev)
                        svd_experts[n].gate_v_proj.weight.data = svd_v.to(dev)
                    elif "down_proj" in name:
                        svd_experts[n].down_u_proj.weight.data = svd_u.to(dev)
                        svd_experts[n].down_v_proj.weight.data = svd_v.to(dev)
                    elif "up_proj" in name:
                        svd_experts[n].up_u_proj.weight.data = svd_u.to(dev)
                        svd_experts[n].up_v_proj.weight.data = svd_v.to(dev)
            # 清理不再使用的变量
            W = None
            del W, svd_u, svd_v
        layer.mlp.experts = svd_experts
        del layer   # 删除当前层的引用
        torch.cuda.empty_cache()
    #print(model)



#======================================#
#   merging
#======================================#
@torch.no_grad()
def calculate_parameter_similarity(expert1, expert2, dev):
    """
    计算两个专家之间的参数相似度（余弦相似度和L2距离）
    
    参数:
    - expert1: 第一个专家模块
    - expert2: 第二个专家模块
    - dev: 设备（例如 'cuda' 或 'cpu'）
    """
    cosine_similarities = []
    l2_distances = []

    def flatten_and_move_to_device(param, device):
        return param.flatten().to(device)

    # 计算 gate_proj 层的相似度
    gate_proj1 = flatten_and_move_to_device(expert1.gate_proj.weight, dev)
    gate_proj2 = flatten_and_move_to_device(expert2.gate_proj.weight, dev)
    cosine_sim_gate = torch.nn.functional.cosine_similarity(gate_proj1.unsqueeze(0), gate_proj2.unsqueeze(0)).item()
    l2_dist_gate = torch.norm(gate_proj1 - gate_proj2).item()
    cosine_similarities.append(cosine_sim_gate)
    l2_distances.append(l2_dist_gate)

    # 计算 up_proj 层的相似度
    up_proj1 = flatten_and_move_to_device(expert1.up_proj.weight, dev)
    up_proj2 = flatten_and_move_to_device(expert2.up_proj.weight, dev)
    cosine_sim_up = torch.nn.functional.cosine_similarity(up_proj1.unsqueeze(0), up_proj2.unsqueeze(0)).item()
    l2_dist_up = torch.norm(up_proj1 - up_proj2).item()
    cosine_similarities.append(cosine_sim_up)
    l2_distances.append(l2_dist_up)

    # 计算 down_proj 层的相似度
    down_proj1 = flatten_and_move_to_device(expert1.down_proj.weight, dev)
    down_proj2 = flatten_and_move_to_device(expert2.down_proj.weight, dev)
    cosine_sim_down = torch.nn.functional.cosine_similarity(down_proj1.unsqueeze(0), down_proj2.unsqueeze(0)).item()
    l2_dist_down = torch.norm(down_proj1 - down_proj2).item()
    cosine_similarities.append(cosine_sim_down)
    l2_distances.append(l2_dist_down)

    # 计算平均值
    avg_cosine_sim = sum(cosine_similarities) / len(cosine_similarities)
    avg_l2_dist = sum(l2_distances) / len(l2_distances)

    return avg_cosine_sim, avg_l2_dist

def evaluate_expert_param_similarity(model_name, model, dev):
    model.eval()  # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    all_layer_parameter_similarities = {}

    for i in range(len(layers)):
        layer = layers[i]
        experts = layer.mlp.experts

        num_experts = len(experts)
        layer_cosine_similarity = torch.zeros((num_experts, num_experts), device=dev)
        layer_l2_distance = torch.zeros((num_experts, num_experts), device=dev)

        for j in range(num_experts):
            for k in range(j, num_experts):
                if j == k:
                    layer_cosine_similarity[j, k] = torch.tensor(1.0, device=dev)
                    layer_l2_distance[j, k] = torch.tensor(0.0, device=dev)
                else:
                    cosine_sim, l2_dist = calculate_parameter_similarity(experts[j], experts[k], dev)
                    layer_cosine_similarity[j, k] = torch.tensor(cosine_sim, device=dev)
                    layer_cosine_similarity[k, j] = torch.tensor(cosine_sim, device=dev)
                    layer_l2_distance[j, k] = torch.tensor(l2_dist, device=dev)
                    layer_l2_distance[k, j] = torch.tensor(l2_dist, device=dev)

        all_layer_parameter_similarities[f'model.layers.{i}.mlp'] = {
            'cosine': layer_cosine_similarity,
            'l2': layer_l2_distance
        }

    return all_layer_parameter_similarities

@torch.no_grad()
def model_expert_merging(model_name, model, expert_importance, expert_similarity, metrics, num_expert_group, dev):
    model.eval()  # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
                
    for i in range(len(layers)):
        layer = layers[i]
        experts = layer.mlp.experts
        gate = layer.mlp.gate  # 假设路由层为 gate

        # 获取该层的专家重要性和相似度矩阵
        layer_importance = expert_importance.get(f'model.layers.{i}.mlp', None)
        layer_similarity = expert_similarity.get(f'model.layers.{i}.mlp', {}).get(metrics, None)

        if layer_importance is None or layer_similarity is None:
            continue

        # 筛选出最重要的 num_expert_group 个主导专家
        sorted_indices = sorted(range(len(layer_importance)), key=lambda k: layer_importance[k], reverse=True)
        leading_experts = sorted_indices[:num_expert_group]

        # 记录每个合并后的专家包含哪些原始专家
        expert_mapping = []
        for leading_expert in leading_experts:
            # 为每个主导专家找到最相似的专家
            similarity_scores = layer_similarity[leading_expert]
            similar_experts = sorted(range(len(similarity_scores)), key=lambda k: similarity_scores[k], reverse=True)
            expert_mapping.append(similar_experts)

        new_experts = []
        for leading_expert in leading_experts:
            # 为每个主导专家找到最相似的专家
            similarity_scores = layer_similarity[leading_expert]
            similar_experts = sorted(range(len(similarity_scores)), key=lambda k: similarity_scores[k], reverse=True)

            # 合并专家
            merged_weight = None
            total_importance = 0
            for expert_idx in similar_experts:
                expert_module = experts[expert_idx]
                importance = layer_importance[expert_idx]
                total_importance += importance
                if hasattr(expert_module, 'weight'):
                    weight = expert_module.weight.data.to(dev) * importance
                    if merged_weight is None:
                        merged_weight = weight
                    else:
                        merged_weight += weight

            # 加权平均
            if total_importance > 0 and merged_weight is not None:
                merged_weight /= total_importance

            # 创建合并后的专家模块
            if merged_weight is not None:
                new_expert = nn.Linear(merged_weight.shape[1], merged_weight.shape[0])
                new_expert.weight.data = merged_weight
            else:
                # 处理其他类型的专家模块，这里简单假设为 Identity 模块
                new_expert = nn.Identity()

            new_experts.append(new_expert)

        # 更新模型的专家模块
        layer.mlp.experts = nn.ModuleList(new_experts).to(dev)
        layer.mlp.num_experts = len(new_experts)

        # 更新路由层
        new_gate_weight = torch.zeros((len(new_experts), gate.weight.shape[1]), device=dev)
        for new_idx, old_experts in enumerate(expert_mapping):
            total_importance = sum(layer_importance[old_idx] for old_idx in old_experts)
            for old_idx in old_experts:
                new_gate_weight[new_idx] += gate.weight.data[old_idx].to(dev) * layer_importance[old_idx]
            if total_importance > 0:
                new_gate_weight[new_idx] /= total_importance

        gate.weight.data = new_gate_weight
        gate.out_features = len(new_experts)

def evaluate_model_expert_importance(model_name, model, tokenizer, calib_loader, device):
    if model_name == 'OLMoE':
        return evaluate_olmoe_expert_importance_low_source(model_name, model, tokenizer, calib_loader, device)
    elif model_name == 'Qwen':
        return evaluate_qwen_expert_importance_low_source(model_name, model, tokenizer, calib_loader, device)

@torch.no_grad()
def evaluate_olmoe_expert_importance_low_source(model_name, model, tokenizer, calib_loader, device):
    # 确保模型仅在需要时加载到 GPU
    model.to(device)
    wrapped_model = OlmoeImportanceWrapper(model, device)
    wrapped_model.eval()
    wrapped_model.to(device)
    wrapped_model.reset_counts()  # 确保统计变量初始化为 CPU 或轻量 GPU 张量

    for batch in tqdm(calib_loader, desc="Calculating expert stats"):
        # 清理前一个 batch 的残留 GPU 内存
        torch.cuda.empty_cache()
        if isinstance(batch, dict):
            batch = [batch]
        # 数据预处理并转移到 GPU
        inputs = prepare_inputs_for_model(batch, tokenizer, device=device)
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}  # 异步传输
        
        # 前向传播
        _ = wrapped_model(**inputs)
        
        # 立即释放输入和中间激活
        del inputs
        torch.cuda.empty_cache()

        # 可选：逐 batch 将统计结果迁移到 CPU（如果包装器支持）
        if hasattr(wrapped_model, 'move_stats_to_cpu'):
            wrapped_model.move_stats_to_cpu()

    # 确保最终统计结果在 CPU 上
    stats = wrapped_model.get_expert_stats()
    if isinstance(stats, dict):
        for key in stats:
            if isinstance(stats[key], torch.Tensor):
                stats[key] = stats[key].cpu()
    wrapped_model.remove_hooks()
    
    # 清理模型和临时变量
    del wrapped_model, model
    torch.cuda.empty_cache()
    
    return stats

@torch.no_grad()
def evaluate_olmoe_expert_importance(model_name, model, tokenizer, calib_loader, device):
    # 确保model和wrapped_model在同一设备上
    wrapped_model = OlmoeImportanceWrapper(model, device)
    wrapped_model.eval()
    wrapped_model.to(device)  # 确保wrapped_model在正确的设备上
    wrapped_model.reset_counts()

    for batch in tqdm(calib_loader, desc="Calculating expert stats"):
        if isinstance(batch, dict):
            batch = [batch]
        
        # 确保prepare_inputs_for_model函数返回的数据也在同一设备上
        inputs = prepare_inputs_for_model(batch, tokenizer, device=device)

        # 确保输入到wrapped_model的数据都在device上
        for key, value in inputs.items():
            inputs[key] = value.to(device)
            
        _ = wrapped_model(**inputs)

    # 获取所有统计信息，并确保它们与使用的设备兼容
    stats = wrapped_model.get_expert_stats()
    wrapped_model.remove_hooks()

    # # 如果plot_expert_statistics涉及任何计算，也需要确保它们在相同的设备上执行
    # plot_expert_statistics(model_name, stats['activation_frequency'], stats['sum_routing_weights'],
    #                        stats['input_l2_avg'], stats['output_l2_avg'])

    return stats

@torch.no_grad()
def plot_expert_statistics(model_name, expert_importance, expert_weight_sum, input_l2_norm_avg, output_l2_norm_avg):
    output_dir = f"outfile/{model_name}/figure"
    os.makedirs(output_dir, exist_ok=True)

    for layer in expert_importance.keys():
        # 绘制专家重要性
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(expert_importance[layer])), expert_importance[layer], tick_label=[str(i) for i in range(len(expert_importance[layer]))])
        plt.title(f"Expert Importance for Layer {layer}")
        plt.xlabel('Expert Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{layer}_expert_frequence.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved expert importance plot for layer {layer} to {filename}")

        # 绘制路由权重累加值
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(expert_weight_sum[layer])), expert_weight_sum[layer], tick_label=[str(i) for i in range(len(expert_weight_sum[layer]))])
        plt.title(f"Routing Weight Sum for Layer {layer}")
        plt.xlabel('Expert Index')
        plt.ylabel('Routing Weight Sum')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{layer}_routing_weight_sum.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved routing weight sum plot for layer {layer} to {filename}")

        # 绘制输入L2范数的平均值
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(input_l2_norm_avg[layer])), input_l2_norm_avg[layer], tick_label=[str(i) for i in range(len(input_l2_norm_avg[layer]))])
        plt.title(f"Average Input L2 Norm for Layer {layer}")
        plt.xlabel('Batch Index')
        plt.ylabel('Input L2 Norm Avg')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{layer}_input_l2_norm_avg.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved average input L2 norm plot for layer {layer} to {filename}")

        # 绘制输出L2范数的平均值
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(output_l2_norm_avg[layer])), output_l2_norm_avg[layer], tick_label=[str(i) for i in range(len(output_l2_norm_avg[layer]))])
        plt.title(f"Average Output L2 Norm for Layer {layer}")
        plt.xlabel('Batch Index')
        plt.ylabel('Output L2 Norm Avg')
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{layer}_output_l2_norm_avg.png")
        plt.savefig(filename)
        plt.close()

        print(f"Saved average output L2 norm plot for layer {layer} to {filename}")

@torch.no_grad()
def evaluate_qwen_expert_importance_low_source(
    model_name: str,
    model: Qwen2MoeForCausalLM,
    tokenizer: AutoTokenizer,
    calib_loader: DataLoader,
    device: str = "cuda"
) -> dict:
    """评估 Qwen-MoE 模型中各专家的重要性指标（显存优化版）"""
    # 初始化模型包装器（默认统计量在 CPU）
    wrapped_model = QwenMoeImportanceWrapper(model, device='cpu')  # 修改包装器内部统计默认在 CPU
    wrapped_model.eval()
    wrapped_model.to(device)
    wrapped_model.reset_counts()  # 确保统计量初始化为 CPU 张量

    # 校准阶段（带显存保护）
    for batch in tqdm(calib_loader, desc="Profiling Expert Stats"):
        try:
            # 清理前序残留显存
            torch.cuda.empty_cache()
            # 生成输入并立即转移至 GPU
            inputs = prepare_inputs_for_qwen(model, batch, tokenizer, device)
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}  # 异步传输
            
            # 前向传播（捕获统计信息）
            _ = wrapped_model(**inputs)
            
            # 立即释放输入和中间激活
            del inputs
            torch.cuda.empty_cache()
            
            # 逐 batch 迁移统计量到 CPU（如果包装器支持）
            if hasattr(wrapped_model, 'flush_stats_to_cpu'):
                wrapped_model.flush_stats_to_cpu()
                
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # 动态回退策略：清空缓存并尝试半精度推理
                torch.cuda.empty_cache()
                with torch.cuda.amp.autocast():
                    _ = wrapped_model(**inputs)
            else:
                raise

    # 确保最终统计量在 CPU 并释放模型
    stats = wrapped_model.get_expert_stats()
    wrapped_model.remove_hooks()
    wrapped_model.to('cpu')  # 确保包装器完全移出 GPU
    del wrapped_model
    torch.cuda.empty_cache()
    
    return stats
@torch.no_grad()
def evaluate_qwen_expert_importance(model_name, model: Qwen2MoeForCausalLM, 
                                  tokenizer: AutoTokenizer,
                                  calib_loader: DataLoader,
                                  device: str = "cuda"):
    # 初始化统计包装
    wrapped_model = QwenMoeImportanceWrapper(model, device)
    wrapped_model.eval()
    wrapped_model.to(device)
    wrapped_model.reset_counts()

    # 校准阶段
    for batch in tqdm(calib_loader, desc="Calculating expert stats"):
        # Qwen专用输入处理
        inputs = prepare_inputs_for_qwen(model, batch, tokenizer, device)
        
        # 前向传播（自动捕获统计信息）
        try:
            outputs = wrapped_model(**inputs)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                half_batch = {k: v[:len(v)//2] for k,v in inputs.items()}
                _ = wrapped_model(**half_batch)
            else:
                raise

    # 获取统计结果
    stats = wrapped_model.get_expert_stats()
    wrapped_model.remove_hooks()
    # plot_expert_statistics(model_name, stats['activation_frequency'], stats['sum_routing_weights'],
    #                        stats['input_l2_avg'], stats['output_l2_avg'])
    
    return stats



@torch.no_grad()
def evaluate_expert_similarity(model, tokenizer, calib_loader, device):
    """
    统计 MoE 模型中每个专家层的 router logits 以及专家相似性指标，包括：
      1. 基于 router logits 的专家相似性（余弦相似性和 L2 距离）。
      2. 基于专家输出的相似性。注意：即使专家没有被选中，其输出仍然为全 0 张量，
         保证输出形状一致，后续计算拼接后各专家输出向量之间的余弦相似性和 L2 距离。
    
    参数:
        model: 要分析的 MoE 模型实例。
        tokenizer: 用于准备输入数据的分词器。
        calib_loader: 校准数据加载器，提供批次数据。
        device: 设备类型（'cuda' 或 'cpu'）。
    
    返回:
        router_logits: 每一层记录的 router logits（字典，键为层名）。
        expert_similarity: 基于 router logits 的专家相似性指标，包含余弦相似性和 L2 距离（字典）。
        output_similarity: 基于专家输出的专家相似性指标，包含余弦相似性和 L2 距离（字典）。
    """
    wrapped_model = OlmoeSimilarityWrapper(model, device)
    wrapped_model.eval()
    wrapped_model.to(device)
    wrapped_model.reset_counts()

    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Calculating expert similarity"):
            if isinstance(batch, dict):
                batch = [batch]
            inputs = prepare_inputs_for_model(batch, tokenizer, device=device)
            _ = wrapped_model(**inputs)

    # 获取 router logits 记录
    #router_logits = wrapped_model.get_router_logits()
    # 计算专家相似性：返回两个字典，一个是基于 router logits 的相似性，另一个是基于专家输出的相似性
    router_logits_similarity, output_similarity = wrapped_model.compute_similarity()

    wrapped_model.remove_hooks()

    return router_logits_similarity, output_similarity

@torch.no_grad()
def evaluate_router_logits_similarity(model, tokenizer, calib_loader, device):
    wrapped_model = OlmoeRouterLogitsSimilarityWrapper(model, device)
    wrapped_model.eval()
    wrapped_model.to(device)
    wrapped_model.reset_counts()

    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Calculating router logits similarity"):
            if isinstance(batch, dict):
                batch = [batch]
            inputs = prepare_inputs_for_model(batch, tokenizer, device=device)
            _ = wrapped_model(**inputs)

    router_logits_similarity = wrapped_model.compute_router_logits_similarity()

    wrapped_model.remove_hooks()

    return router_logits_similarity

def evaluate_output_similarity(model, tokenizer, calib_loader, device):
    wrapped_model = OlmoeOutputLogitsSimilarityWrapper(model, device)
    wrapped_model.eval()
    wrapped_model.to(device)
    wrapped_model.reset_counts()

    with torch.no_grad():
        for batch in tqdm(calib_loader, desc="Calculating expert similarity"):
            if isinstance(batch, dict):
                batch = [batch]
            inputs = prepare_inputs_for_model(batch, tokenizer, device=device)
            _ = wrapped_model(**inputs)

    expert_output_similarity = wrapped_model.compute_expert_similarity()

    wrapped_model.remove_hooks()

    return expert_output_similarity



#=====================================#
#  pruning
##=====================================#
import logging
from component.modeling_pruning import PrunableOlmoeSparseMoeBlockWrapper
from component.modeling_compressqwen import PrunableQwenMoeSparseMoeBlockWrapper
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
logger = logging.getLogger(__name__)

def model_expert_pruning(dev, model_name, model, experts_importance, strategy='threshold', all_expert_pruning_num=None):
    if model_name == 'OLMoE':
        return olmoe_expert_pruning(dev, model, experts_importance, strategy, all_expert_pruning_num)
    elif model_name == 'Qwen':
        return qwen_expert_pruning(dev, model, experts_importance, strategy, all_expert_pruning_num)

def olmoe_expert_pruning(dev, model: OlmoeForCausalLM, experts_importance,
                         strategy='threshold', all_expert_pruning_num=None):
    """
    对 Olmoe 模型的每一层进行专家剪枝，支持两种剪枝策略：
    1. 'threshold'：剪去重要性较低的 all_expert_pruning_num 个专家；
    2. 'fixed'：每层尽量均匀地剪去一定数量的专家，使得总剪枝数量达到 all_expert_pruning_num。

    同时记录下每一层剪枝后保留的专家的原始序号，存储在 all_original_expert_indices 中，并一并返回。
    """
    num_layers = len(model.model.layers)
    num_remained_per_layer = []

    if strategy == 'threshold':
        if all_expert_pruning_num is None:
            raise ValueError("For threshold strategy, all_expert_pruning_num must be provided.")
        # 收集所有专家的重要度和对应的层索引、专家索引
        all_experts_info = []
        for l, layer in enumerate(model.model.layers):
            current_layer_experts_importance = experts_importance.get(f'model.layers.{l}.mlp', None)
            if current_layer_experts_importance is None:
                logger.warning(f"No importance scores found for layer {l}. Skipping this layer.")
                continue
            for e, importance in enumerate(current_layer_experts_importance):
                all_experts_info.append((l, e, importance))

        # 按重要度排序
        all_experts_info.sort(key=lambda x: x[2])

        # 初始化每层剪枝数量
        prune_count_per_layer = [0] * num_layers
        total_pruned = 0
        for i in range(len(all_experts_info)):
            if total_pruned >= all_expert_pruning_num:
                break
            layer_index = all_experts_info[i][0]
            if len(model.model.layers[layer_index].mlp.experts) - prune_count_per_layer[layer_index] > 1:
                prune_count_per_layer[layer_index] += 1
                total_pruned += 1

        # 计算每层剩余专家数量
        for l, layer in enumerate(model.model.layers):
            current_layer_experts = len(layer.mlp.experts)
            num_remained_per_layer.append(current_layer_experts - prune_count_per_layer[l])

    elif strategy == 'fixed':
        if all_expert_pruning_num is None:
            raise ValueError("For fixed strategy, all_expert_pruning_num must be provided.")

        # 计算每层平均要剪枝的专家个数
        avg_prune_count = all_expert_pruning_num // num_layers
        remainder = all_expert_pruning_num % num_layers

        for l, layer in enumerate(model.model.layers):
            current_layer_experts = len(layer.mlp.experts)
            # 分配剪枝个数，余数依次分配到前面的层
            prune_count = avg_prune_count
            if remainder > 0:
                prune_count += 1
                remainder -= 1
            # 确保每层至少保留一个专家
            if current_layer_experts - prune_count < 1:
                prune_count = current_layer_experts - 1
            num_remained_per_layer.append(current_layer_experts - prune_count)
    else:
        raise ValueError("Unknown pruning strategy. Use 'threshold' or 'fixed'.")

    print('num_remained_per_layer', num_remained_per_layer)
    assert isinstance(model, OlmoeForCausalLM), 'Currently only `Olmoe` is supported'

    # 将每层的 mlp 包装成可剪枝的 wrapper，并打开缓存选项
    for l, layer in enumerate(model.model.layers):
        layer.mlp = PrunableOlmoeSparseMoeBlockWrapper(layer.mlp, r=None)  # 不需要传递固定的 r 值
        layer.mlp.cache_X = True
        layer.mlp.cache_Z = True

    model.to(dev)  # 确保模型移动到指定设备
    torch.cuda.empty_cache()

    global_loss_history = dict()
    all_original_expert_indices = {}  # 用于记录所有层剪枝后保留的原始专家索引

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Pruning based on importance scores...'):
        b = layer.mlp
        if not hasattr(b, 'cache_space'):
            continue

        # 获取当前层专家的重要性分数
        current_layer_experts_importance = experts_importance.get(f'model.layers.{l}.mlp', None)
        if current_layer_experts_importance is None:
            logger.warning(f"No importance scores found for layer {l}. Skipping this layer.")
            continue

        b.prune_by_importance(current_layer_experts_importance, num_remained_per_layer[l])

        # 记录该层剪枝后保留的原始专家索引
        all_original_expert_indices[f'layers.{l}.mlp'] = b.original_expert_indices

        # 调整 top_k 保证不超过当前剩余的专家数量
        b.model.top_k = min(b.model.top_k, b.model.num_experts)
        b.to(dev)  # 确保 b 移动到指定设备

    for l, layer in enumerate(model.model.layers):
        layer.mlp = layer.mlp.model

    # 更新模型配置中专家的数量
    model.num_experts = sum(len(layer.mlp.experts) for layer in model.model.layers)
    model.config.num_local_experts = model.num_experts 

    # 返回模型、global_loss_history 以及所有层剪枝后保留的原始专家索引
    return model

def qwen_expert_pruning(dev, model: Qwen2MoeForCausalLM, experts_importance,
                       strategy='threshold', all_expert_pruning_num=None):
    """
    修复后的Qwen-MoE剪枝函数
    关键修改点：
    1. 适配专家重要性数据结构为{layer_name: [expert_scores]}
    2. 显式处理共享专家索引（假设每层最后一个专家是共享专家）
    """
    num_layers = len(model.model.layers)
    num_remained_per_layer = []
    all_original_expert_indices = {}

    # 阶段1：计算各层应保留的专家数
    if strategy == 'threshold':
        # 收集所有可剪枝专家信息（排除共享专家）
        all_experts_info = []
        for l in range(num_layers):
            layer_key = f'model.layers.{l}.mlp'
            expert_scores = experts_importance.get(layer_key, [])
            # 假设共享专家是最后一个，且不被剪枝
            num_regular = len(expert_scores) - 1  # 排除共享专家
            for e in range(num_regular):
                all_experts_info.append( (l, e, expert_scores[e]) )

        # 按重要性排序并分配剪枝名额
        all_experts_info.sort(key=lambda x: x[2])
        prune_counts = [0]*num_layers
        total_pruned = 0
        
        for expert_info in all_experts_info:
            if total_pruned >= all_expert_pruning_num:
                break
            l, e, _ = expert_info
            current_layer = model.model.layers[l].mlp
            # 确保每层至少保留1个普通专家
            if (len(current_layer.experts) - 1 - prune_counts[l]) > 1:  # -1排除共享专家
                prune_counts[l] += 1
                total_pruned += 1
        
        num_remained_per_layer = [
            (len(layer.mlp.experts) - 1) - prune_counts[l] + 1  # 保留共享专家
            for l, layer in enumerate(model.model.layers)
        ]

    elif strategy == 'fixed':
        # 计算平均每层剪枝数（排除共享专家）
        total_prunable = sum(
            len(layer.mlp.experts) - 1  # 排除共享专家
            for layer in model.model.layers
        )
        if all_expert_pruning_num > total_prunable:
            raise ValueError(f"Total pruned number need < {total_prunable}")

        avg_prune = all_expert_pruning_num // num_layers
        remainder = all_expert_pruning_num % num_layers

        for l, layer in enumerate(model.model.layers):
            max_prune = len(layer.mlp.experts) - 1  # 最多剪枝数（保留共享专家）
            layer_prune = avg_prune + (1 if l < remainder else 0)
            actual_prune = min(layer_prune, max_prune)
            num_remained_per_layer.append(
                (len(layer.mlp.experts) - 1) - actual_prune + 1  # 保留共享专家
            )

    else:
        raise ValueError("Unkonw Strategy")

    # 阶段2：初始化可剪枝包装
    for l, layer in enumerate(model.model.layers):
        original_mlp = layer.mlp
        layer.mlp = PrunableQwenMoeSparseMoeBlockWrapper(original_mlp)
        layer.mlp.cache_X = True
        layer.mlp.cache_Z = True

    model.to(dev)
    torch.cuda.empty_cache()

    # 阶段3：逐层执行剪枝
    for l in tqdm(range(num_layers), desc="Pruning based on importance scores..."):
        layer = model.model.layers[l]
        if not isinstance(layer.mlp, PrunableQwenMoeSparseMoeBlockWrapper):
            continue

        # 获取当前层重要性分数（排除共享专家）
        layer_key = f'model.layers.{l}.mlp'
        expert_scores = experts_importance.get(layer_key, [])
        if len(expert_scores) == 0:
            logger.warning(f"Overjump{l}layer:no importance scores")
            continue
        
        # 假设最后一个分数对应共享专家
        regular_scores = expert_scores[:-1]  # 普通专家分数
        shared_score = expert_scores[-1]     # 共享专家分数

        target_remain = num_remained_per_layer[l] - 1  # 保留的普通专家数
        if target_remain <= 0:
            raise RuntimeError(f"Layer{l} experts number < 0")

        # 执行剪枝（仅处理普通专家）
        layer.mlp.prune_by_importance(
            experts_importance=regular_scores,
            num_experts_remain=target_remain
        )
        
        # 记录原始索引（保留共享专家）
        pruned_indices = layer.mlp.original_expert_indices
        all_original_expert_indices[f'layers.{l}.mlp'] = (
            pruned_indices + [len(regular_scores)]  # 添加共享专家索引
        )

    # 阶段4：恢复原始模型结构
    for l in range(num_layers):
        layer = model.model.layers[l]
        if isinstance(layer.mlp, PrunableQwenMoeSparseMoeBlockWrapper):
            layer.mlp = layer.mlp.model  # 解除包装

    # 更新模型配置
    model.config.num_experts = sum(
        len(layer.mlp.experts) 
        for layer in model.model.layers
    )
    logger.info(f"Finish pruning ,number of experts remained：{model.config.num_experts}")

    return model, all_original_expert_indices


# 三种压缩方法的入口

@torch.no_grad()
def step_pruning(args, model, tokenizer, device, num_experts_pruning):
    if (not getattr(args.pruning, "enabled", False)) or num_experts_pruning <= 0:
        return model, None

    cali_data = get_calib_train_data(
        args.dataset,
        tokenizer,
        args.pruning.eval_nsamples,
        args.eval.model_seq_len,
        args.seed,
    )
    importance_data = evaluate_model_expert_importance(args.model_name, model, tokenizer, cali_data, device)
    experts_importance = importance_data[f"{args.pruning.importance_metrics}"]

    model = model_expert_pruning(
        device,
        args.model_name,
        model,
        experts_importance,
        args.pruning.strategy,
        num_experts_pruning,
    )

    # 释放中间变量
    del cali_data, importance_data, experts_importance
    return model


@torch.no_grad()
def step_merging(args, model, tokenizer, device, num_expert_group):
    if (not getattr(args.merging, "enabled", False)) or num_expert_group <= 0:
        return model

    # 1) 计算 similarity（基于权重 或 基于输出/router logits）
    if args.merging.eval_object == "weight":
        expert_similarity = evaluate_expert_param_similarity(args.model_name, model, device)
    else:
        cali_data = get_calib_train_data(
            args.dataset,
            tokenizer,
            args.merging.eval_nsamples,
            args.eval.model_seq_len,
            args.seed,
        )
        router_logits_similarity, output_similarity = evaluate_expert_similarity(model, tokenizer, cali_data, device)
        del cali_data
        if args.merging.eval_object == "output":
            expert_similarity = output_similarity
            del router_logits_similarity
        else:
            expert_similarity = router_logits_similarity
            del output_similarity

    # 2) 计算 weighting_factor（注意：这里用 weighting_factor_data，不要用已经被删的 expert_importance_data）
    cali_data = get_calib_train_data(
        args.dataset,
        tokenizer,
        args.merging.eval_nsamples,
        args.eval.model_seq_len,
        args.seed,
    )
    weighting_factor_data = evaluate_model_expert_importance(args.model_name, model, tokenizer, cali_data, device)
    weighting_factor = weighting_factor_data[f"{args.merging.weighting_factor}"]
    del cali_data, weighting_factor_data

    # 3) 执行 merging
    model_expert_merging(
        args.model_name,
        model,
        weighting_factor,
        expert_similarity,
        args.merging.metrics,
        num_expert_group,
        device,
    )

    del weighting_factor, expert_similarity
    return model


@torch.no_grad()
def step_decomposition(args, model, tokenizer, device, lora_rank):
    if (not getattr(args.svd, "enabled", False)) or lora_rank <= 0:
        return model

    profiling_mat = None
    if args.svd.method != "vanilla-SVD":
        cali_data = get_calib_train_data(
            args.dataset,
            tokenizer,
            args.svd.whitening_nsamples,
            seqlen=args.eval.model_seq_len,
        )
        if args.svd.method == "SVD-LLM":
            profiling_mat = profile_svdllm(model, cali_data, device)
        elif args.svd.method == "ASVD":
            profiling_mat = profile_asvd(model, cali_data, device, alpha=0.5)
        else:
            raise ValueError(f"Unknown svd.method={args.svd.method}")
        del cali_data

    model_expert_only_SVD(args.model_name, model, args.svd.method, lora_rank, device, profiling_mat)
    del profiling_mat
    return model


@torch.no_grad()
def apply_compression_by_order(args, model, tokenizer, device,
                               order, num_experts_pruning, num_expert_group, lora_rank, param_ratio):
    """
    order: ["P","M","D"] 的任意排列/子序列
    """

    for op in order:
        if op == "P" and param_ratio["pruning_ratio"]!=0:
            model = step_pruning(
                args, model, tokenizer, device, num_experts_pruning
            )
        elif op == "M" and param_ratio["merging_ratio"]!=0:
            model = step_merging(
                args, model, tokenizer, device, num_expert_group
            )
        elif op == "D" and param_ratio["svd_ratio"]!=0:
            model = step_decomposition(
                args, model, tokenizer, device, lora_rank
            )

        # 每步之后适当清理
        torch.cuda.empty_cache()
        gc.collect()

    return model