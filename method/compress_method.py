#coding:utf8
import os
from tqdm import tqdm
import torch
import torch.nn as nn

from ..component.modeling_compressdeepseek import SVD_DeepseekMLP
from ..component.modeling_merging import *
from ..component.modeling_compressolmoe import OlmoeImportanceWrapper,SVD_OlmoeMLP
from ..component.modeling_compressqwen import QwenMoeImportanceWrapper, SVD_Qwen2MoeMLP
from ..utils.model_utils import find_layers, prepare_inputs_for_model, prepare_inputs_for_qwen

import matplotlib.pyplot as plt
from expert_pruning import *
from torch.utils.data import DataLoader

# 模型相关
from transformers import (
    AutoTokenizer,
    Qwen2MoeForCausalLM
)
import numpy as np

#=============SVD================

@torch.no_grad()
def profile_svdllm_low_source(model, calib_loader, dev):
    layers = model.model.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")

    # 定义前向钩子，在CPU上累加统计矩阵
    def hook(module, input, output):
        inp = input[0].detach().float()  # 保持在GPU上计算以提高速度
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        # 计算并转移到CPU
        adds = torch.matmul(inp.transpose(1, 2), inp)  # [batch_size, in_features, in_features]
        adds_sum = torch.sum(adds, dim=0).cpu()  # 结果转移到CPU
        module.raw_scaling_diag_matrix += adds_sum  # 在CPU上累加
        del inp, adds, adds_sum
        torch.cuda.empty_cache()

    # 初始化统计矩阵在CPU上
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            module.raw_scaling_diag_matrix = torch.zeros((in_features, in_features), device='cpu')
            module.register_forward_hook(hook)

    # 前向传播收集统计信息
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
        torch.cuda.empty_cache()  # 每个batch后清理显存

    # 清理钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()

    profiling_mat = {}
    # 逐层处理，避免同时保存所有层数据
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            # 从CPU加载统计矩阵并转换到GPU计算
            raw_cpu = subset[name].raw_scaling_diag_matrix
            raw_gpu = raw_cpu.to(dev)
            
            # 确保矩阵正定
            try:
                scaling_matrix = torch.linalg.cholesky(raw_gpu)
            except RuntimeError:
                # 添加正则化项
                eigenvalues = torch.linalg.eigvalsh(raw_gpu)
                regularization = (-eigenvalues[0] + 1e-6).item()
                raw_gpu += regularization * torch.eye(raw_gpu.size(0), device=dev)
                scaling_matrix = torch.linalg.cholesky(raw_gpu)
                del eigenvalues
            
            # 结果移回CPU并保存
            layer_profile[name] = scaling_matrix.cpu()
            # 释放GPU显存
            del raw_gpu, scaling_matrix
            subset[name].raw_scaling_diag_matrix = None  # 及时释放CPU内存
            torch.cuda.empty_cache()
        
        profiling_mat[i] = layer_profile
        # 释放当前层的资源
        del subset
        torch.cuda.empty_cache()
    
    return profiling_mat

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
def profile_asvd_low_source(model, calib_loader, dev, alpha=0.5):
    layers = model.model.layers
    model = model.to(dev)
    print("Start obtaining the scaling matrix S...")

    # 优化点1：使用弱引用容器管理钩子
    hooks = []

    def hook(module, input, output):
        try:
            # 优化点2：原地操作+立即释放内存
            inp = input[0].detach()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            
            # 优化点3：融合计算流程
            with torch.no_grad():
                # 原地绝对值计算
                inp_abs = inp.abs_()  # 减少内存分配
                
                # 直接计算通道维度均值
                abs_mean = torch.mean(inp_abs, dim=(0, 1), keepdim=False)
                
                # 原地累加统计量
                module.raw_scaling_diag_matrix.add_(abs_mean)
                
            # 显式释放中间变量
            del inp, inp_abs, abs_mean
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                print(f"Warning: 触发显存回收机制，当前batch跳过统计")
            else:
                raise

    # 初始化全精度统计矩阵
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 优化点4：延迟分配显存
            module.register_buffer('raw_scaling_diag_matrix', 
                                 torch.zeros(module.weight.size(1), device=dev))
            hooks.append(module.register_forward_hook(hook))

    # 校准阶段显存保护
    try:
        for batch in tqdm(calib_loader, desc="Profiling"):
            # 优化点5：分项迁移+立即释放
            batch_on_dev = {}
            for k in list(batch.keys()):  # 遍历时动态删除
                batch_on_dev[k] = batch[k].to(dev, non_blocking=True)
                del batch[k]  # 立即释放CPU数据
                
            model(**batch_on_dev)
            
            del batch_on_dev
            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("提前终止校准，使用部分统计结果")

    # 显式清理钩子
    for h in hooks:
        h.remove()
    del hooks
    torch.cuda.empty_cache()

    # 计算阶段优化
    profiling_mat = {}
    print("Start computing the scaling matrix S...")
    for i in tqdm(range(len(layers)), desc="Processing Layers"):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            module = subset[name]
            
            # 优化点6：直接操作+就地计算
            abs_mean_activations = module.raw_scaling_diag_matrix
            
            # 数值稳定性处理（合并操作）
            torch.nan_to_num(
                abs_mean_activations, 
                nan=1e-6, 
                posinf=1e6,
                neginf=0,
                out=abs_mean_activations  # 原地修改
            )
            
            # 生成对角矩阵（保持完整矩阵形式）
            Sii = torch.pow(abs_mean_activations, alpha)
            scaling_diag_matrix = torch.diag(Sii)
            
            # 优化点7：延迟矩阵生成
            layer_profile[name] = scaling_diag_matrix.detach().cpu()
            
            # 立即清理显存
            del abs_mean_activations, Sii, scaling_diag_matrix
            module._buffers.pop('raw_scaling_diag_matrix', None)
            torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile

    return profiling_mat


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
                #print(f"Layer {i}, Module {name}: abs_mean_activations 包含 NaN 或 inf 值")
                abs_mean_activations[torch.isnan(abs_mean_activations)] = 0
                abs_mean_activations[torch.isinf(abs_mean_activations)] = 0

            # Compute Sii based on the average magnitude of activation in each channel
            Sii = (abs_mean_activations ** alpha)
            
            # Ensure no non-finite values in Sii
            if contains_inf_or_nan(Sii):
                #print(f"Layer {i}, Module {name}: Sii 包含 NaN 或 inf 值")
                Sii[torch.isnan(Sii)] = 1e-6  # Small positive value
                Sii[torch.isinf(Sii)] = 1e-6  # Small positive value

            scaling_diag_matrix = torch.diag(Sii)

            # Ensure no non-finite values in scaling_diag_matrix
            if contains_inf_or_nan(scaling_diag_matrix):
                #print(f"Layer {i}, Module {name}: scaling_diag_matrix 包含 NaN 或 inf 值")
                scaling_diag_matrix[torch.isnan(scaling_diag_matrix)] = 1e-6
                scaling_diag_matrix[torch.isinf(scaling_diag_matrix)] = 1e-6

            layer_profile[name] = scaling_diag_matrix
            del abs_mean_activations, Sii, scaling_diag_matrix
            torch.cuda.empty_cache()

        profiling_mat[i] = layer_profile

    return profiling_mat

@torch.no_grad()
def calculate_parameter_similarity_old(expert1, expert2, dev):
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
@torch.no_grad()
def calculate_parameter_similarity(expert1, expert2, dev):
    """
    计算两个专家之间的参数相似度（余弦相似度和 L2 距离）
    
    参数:
    - expert1: 第一个专家模块 (可以是 OlmoeMLP 或 SVD_OlmoeMLP)
    - expert2: 第二个专家模块 (可以是 OlmoeMLP 或 SVD_OlmoeMLP)
    - dev: 设备（例如 'cuda' 或 'cpu'）
    """
    cosine_similarities = []
    l2_distances = []

    def flatten_and_move_to_device(param, device):
        return param.flatten().to(device)

    # 定义一个通用函数，用于计算单层参数的相似度
    def compute_similarity_for_layer(param1, param2, device):
        param1_flat = flatten_and_move_to_device(param1, device)
        param2_flat = flatten_and_move_to_device(param2, device)
        cosine_sim = torch.nn.functional.cosine_similarity(
            param1_flat.unsqueeze(0), param2_flat.unsqueeze(0)
        ).item()
        l2_dist = torch.norm(param1_flat - param2_flat).item()
        return cosine_sim, l2_dist

    # 定义一个辅助函数，用于处理 SVD 分解后的低秩矩阵
    def compute_similarity_for_svd_layer(proj1_u, proj1_v, proj2_u, proj2_v, device):
        # 将 u_proj 和 v_proj 展平并移动到指定设备
        proj1_u_flat = flatten_and_move_to_device(proj1_u.weight, device)
        proj1_v_flat = flatten_and_move_to_device(proj1_v.weight, device)
        proj2_u_flat = flatten_and_move_to_device(proj2_u.weight, device)
        proj2_v_flat = flatten_and_move_to_device(proj2_v.weight, device)

        # 合并 u_proj 和 v_proj 的参数
        proj1_params = torch.cat([proj1_u_flat, proj1_v_flat])
        proj2_params = torch.cat([proj2_u_flat, proj2_v_flat])

        # 计算余弦相似度和 L2 距离
        cosine_sim = torch.nn.functional.cosine_similarity(
            proj1_params.unsqueeze(0), proj2_params.unsqueeze(0)
        ).item()
        l2_dist = torch.norm(proj1_params - proj2_params).item()

        return cosine_sim, l2_dist

    # 动态判断专家模块的类型
    if hasattr(expert1, 'gate_proj') and hasattr(expert2, 'gate_proj'):
        # 处理正常的 OlmoeMLP 结构
        layers = ['gate_proj', 'up_proj', 'down_proj']
        for layer in layers:
            param1 = getattr(expert1, layer).weight
            param2 = getattr(expert2, layer).weight
            cosine_sim, l2_dist = compute_similarity_for_layer(param1, param2, dev)
            cosine_similarities.append(cosine_sim)
            l2_distances.append(l2_dist)
    elif hasattr(expert1, 'gate_u_proj') and hasattr(expert2, 'gate_u_proj'):
        # 处理 SVD_OlmoeMLP 结构
        cosine_sim_gate, l2_dist_gate = compute_similarity_for_svd_layer(
            expert1.gate_u_proj, expert1.gate_v_proj,
            expert2.gate_u_proj, expert2.gate_v_proj, dev
        )
        cosine_sim_up, l2_dist_up = compute_similarity_for_svd_layer(
            expert1.up_u_proj, expert1.up_v_proj,
            expert2.up_u_proj, expert2.up_v_proj, dev
        )
        cosine_sim_down, l2_dist_down = compute_similarity_for_svd_layer(
            expert1.down_u_proj, expert1.down_v_proj,
            expert2.down_u_proj, expert2.down_v_proj, dev
        )
        cosine_similarities.extend([cosine_sim_gate, cosine_sim_up, cosine_sim_down])
        l2_distances.extend([l2_dist_gate, l2_dist_up, l2_dist_down])
    else:
        raise ValueError("Unsupported expert module type!")

    # 计算平均值
    avg_cosine_sim = sum(cosine_similarities) / len(cosine_similarities)
    avg_l2_dist = sum(l2_distances) / len(l2_distances)

    return avg_cosine_sim, avg_l2_dist
def evaluate_expert_param_similarity(model_name, model, dev):
    model.eval()  # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    model.to(dev)  # 确保模型在指定设备上
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    all_layer_parameter_similarities = {}

    for i in range(len(layers)):
        layer = layers[i]
        experts = layer.mlp.experts

        num_experts = len(experts)
        if num_experts < 2:
            continue  # 至少需要两个专家才能计算相似度
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


from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
    
def slerp(v0, v1, t):
    """球面线性插值（SLERP）"""
    v0 = F.normalize(v0, p=2, dim=0)
    v1 = F.normalize(v1, p=2, dim=0)
    dot = (v0 * v1).sum().clamp(-1, 1)
    theta = dot.acos()
    sin_theta = theta.sin()
    if sin_theta < 1e-6:  # 处理同方向向量
        return (1 - t) * v0 + t * v1
    return (v0 * (theta * (1 - t)).sin() + v1 * (theta * t).sin()) / sin_theta


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


# def slerp(v0, v1, t):
#     """球面线性插值（SLERP）"""
#     v0 = F.normalize(v0, p=2, dim=0)
#     v1 = F.normalize(v1, p=2, dim=0)
#     dot = (v0 * v1).sum().clamp(-1, 1)
#     theta = dot.acos()
#     sin_theta = theta.sin()
#     if sin_theta < 1e-6:  # 处理同方向向量
#         return (1 - t) * v0 + t * v1
#     return (v0 * (theta * (1 - t)).sin() + v1 * (theta * t).sin()) / sin_theta

# def uniform_slerp(vectors):
#     """均匀球面插值"""
#     result = vectors[0].clone()
#     for i in range(1, len(vectors)):
#         result = slerp(result, vectors[i], 1 / (i + 1))
#     return result

def slerp(a, b, t):
    """
    球面线性插值函数
    :param a: 第一个向量
    :param b: 第二个向量
    :param t: 插值参数，范围 [0, 1]
    :return: 插值后的向量
    """
    omega = torch.acos(torch.clamp(torch.dot(a.view(-1), b.view(-1)) / (torch.norm(a) * torch.norm(b)), -1, 1))
    sin_omega = torch.sin(omega)
    if sin_omega < 1e-5:
        # 当夹角很小时，退化为线性插值
        return (1 - t) * a + t * b
    return (torch.sin((1 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


def uniform_slerp(vectors):
    """
    均匀球面插值，对多个向量进行球面插值合并
    :param vectors: 向量列表
    :return: 合并后的向量
    """
    result = vectors[0].clone()
    for i in range(1, len(vectors)):
        result = slerp(result, vectors[i], 1 / (i + 1))
    return result

@torch.no_grad()
def model_expert_merging_leading(model_name, model, expert_importance, expert_similarity, metrics, num_expert_group, dev, all_original_expert_indices=None):
    model.eval()
    print(num_expert_group)
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # 计算各层的组数分配
    num_layers = len(layers)
    base_groups_per_layer = num_expert_group // num_layers
    remainder_groups = num_expert_group % num_layers
    layer_groups = [base_groups_per_layer] * num_layers
    # 从最后一层开始分配余数
    for i in range(num_layers - 1, num_layers - remainder_groups - 1, -1):
        layer_groups[i] += 1

    print(layer_groups)
    for i in range(len(layers)):
        layer = layers[i]
        if all_original_expert_indices is not None:
            layer_original_expert_indices = all_original_expert_indices.get(f'layers.{i}.mlp', None)
        experts = layer.mlp.experts
        gate = layer.mlp.gate

        # 获取该层的专家重要性和相似度矩阵
        layer_importance = expert_importance.get(f'model.layers.{i}.mlp', None)
        layer_similarity = expert_similarity.get(f'model.layers.{i}.mlp', {}).get(metrics, None)
        num_groups_per_layer = layer_groups[i]

        # 根据原始索引调整重要性和相似度矩阵
        if all_original_expert_indices is None:
            adjusted_layer_importance = layer_importance
            adjusted_layer_similarity = layer_similarity
        else:
            adjusted_layer_importance = [layer_importance[idx] for idx in layer_original_expert_indices]
            adjusted_layer_similarity = [[layer_similarity[i][j] for j in layer_original_expert_indices]
                                         for i in layer_original_expert_indices]

        total_experts = len(adjusted_layer_importance)

        # 确定主导专家
        sorted_importance = sorted(enumerate(adjusted_layer_importance), key=lambda x: x[1], reverse=True)
        leading_experts = [idx for idx, _ in sorted_importance[:num_groups_per_layer]]
        non_leading_experts = [idx for idx in range(total_experts) if idx not in leading_experts]

        # 初始化专家分组
        expert_mapping = [[] for _ in range(num_groups_per_layer)]
        for idx in leading_experts:
            group_index = leading_experts.index(idx)
            expert_mapping[group_index].append(idx)

        # 为非主导专家分配到主导专家组
        for non_leading_idx in non_leading_experts:
            similarities = [adjusted_layer_similarity[non_leading_idx][leading_idx] for leading_idx in leading_experts]
            if metrics == 'cosine':
                most_similar_group_index = similarities.index(max(similarities))
            elif metrics == 'l2':
                most_similar_group_index = similarities.index(min(similarities))
            expert_mapping[most_similar_group_index].append(non_leading_idx)

        # 合并专家参数
        new_experts = []
        for group in expert_mapping:
            if not group:
                continue

            # 初始化合并参数
            merged_params = {
                'gate_proj': [],
                'up_proj': [],
                'down_proj': []
            }

            # 收集所有专家的权重
            for expert_idx in group:
                expert = experts[expert_idx].to(dev)
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    w = getattr(expert, proj).weight.data.clone()
                    merged_params[proj].append(w)

            # 合并权重：根据 metrics 选择不同的合并方式
            if metrics == 'cosine':
                # 使用均匀球面插值合并
                merged_params = {
                    proj: uniform_slerp(weights) for proj, weights in merged_params.items()
                }
            else:
                # 使用简单的平均加权合并
                merged_params = {
                    proj: torch.mean(torch.stack(weights), dim=0) for proj, weights in merged_params.items()
                }

            # 创建新专家
            new_expert = OlmoeMLP(model.config).to(dev)
            new_expert.gate_proj.weight.data = merged_params['gate_proj']
            new_expert.up_proj.weight.data = merged_params['up_proj']
            new_expert.down_proj.weight.data = merged_params['down_proj']
            new_experts.append(new_expert)

        # 更新路由层参数
        if new_experts:
            layer.mlp.experts = nn.ModuleList(new_experts)
            layer.mlp.num_experts = len(new_experts)
            # 更新门控权重
            new_gate_weight = torch.zeros((len(new_experts), gate.weight.shape[1]), device=dev)
            for new_idx, group in enumerate(expert_mapping):
                group_weights = torch.stack([gate.weight.data[old_idx].to(dev) for old_idx in group])
                new_gate_weight[new_idx] = group_weights.mean(dim=0)

            gate.weight = nn.Parameter(new_gate_weight)
            gate.out_features = len(new_experts)
            layer.mlp.top_k = min(layer.mlp.top_k, len(new_experts))

@torch.no_grad()
def model_expert_merging_clustering(model_name, model, expert_importance, expert_similarity, metrics, num_expert_group, dev, all_original_expert_indices=None):
    model.eval()
    print(num_expert_group)
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # 计算各层的组数分配
    num_layers = len(layers)
    base_groups_per_layer = num_expert_group // num_layers
    remainder_groups = num_expert_group % num_layers
    layer_groups = [base_groups_per_layer] * num_layers
    # 从最后一层开始分配余数
    for i in range(num_layers - 1, num_layers - remainder_groups - 1, -1):
        layer_groups[i] += 1

    print(layer_groups)
    for i in range(len(layers)):
        layer = layers[i]
        if all_original_expert_indices is not None:
            layer_original_expert_indices = all_original_expert_indices.get(f'layers.{i}.mlp', None)
            print(layer_original_expert_indices)
        experts = layer.mlp.experts
        gate = layer.mlp.gate

        # 获取该层的专家重要性和相似度矩阵
        layer_importance = expert_importance.get(f'model.layers.{i}.mlp', None)
        layer_similarity = expert_similarity.get(f'model.layers.{i}.mlp', {}).get(metrics, None)
        num_groups_per_layer = layer_groups[i]

        # 根据原始索引调整重要性和相似度矩阵
        if all_original_expert_indices is None:
            adjusted_layer_importance = layer_importance
            adjusted_layer_similarity = layer_similarity
        else:
            adjusted_layer_importance = [layer_importance[idx] for idx in layer_original_expert_indices]
            adjusted_layer_similarity = [[layer_similarity[i][j] for j in layer_original_expert_indices]
                                         for i in layer_original_expert_indices]

        total_experts = len(adjusted_layer_importance)
        # 根据相似度进行层次聚类
        clusters = [[i] for i in range(total_experts)]
        while len(clusters) > num_groups_per_layer:
            best_pair = None
            best_value = None
            # 遍历所有可能的簇对
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 计算当前两个簇之间的相似度或距离
                    current_value = None
                    if metrics == 'cosine':
                        # 计算两个簇之间的最大相似度
                        max_sim = -float('inf')
                        for x in clusters[i]:
                            for y in clusters[j]:
                                sim = adjusted_layer_similarity[x][y]
                                if sim > max_sim:
                                    max_sim = sim
                        current_value = max_sim
                    elif metrics == 'l2':
                        # 计算两个簇之间的最小距离
                        min_dist = float('inf')
                        for x in clusters[i]:
                            for y in clusters[j]:
                                dist = adjusted_layer_similarity[x][y]
                                if dist < min_dist:
                                    min_dist = dist
                        current_value = min_dist
                    # 更新最佳对
                    if metrics == 'cosine':
                        if best_value is None or current_value > best_value:
                            best_value = current_value
                            best_pair = (i, j)
                    elif metrics == 'l2':
                        if best_value is None or current_value < best_value:
                            best_value = current_value
                            best_pair = (i, j)
            # 合并最佳对
            if best_pair is not None:
                i, j = best_pair
                clusters[i].extend(clusters[j])
                del clusters[j]
            else:
                break  # 处理无法合并的情况

        expert_mapping = clusters

        # 合并专家参数
        new_experts = []
        for group in expert_mapping:
            if not group:
                continue

            # 初始化合并参数
            merged_params = {
                'gate_proj': [],
                'up_proj': [],
                'down_proj': []
            }

            # 收集所有专家的权重
            for expert_idx in group:
                expert = experts[expert_idx].to(dev)
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    w = getattr(expert, proj).weight.data.clone()
                    merged_params[proj].append(w)

            # 合并权重：根据 metrics 选择不同的合并方式
            if metrics == 'cosine':
                # 使用均匀球面插值合并
                merged_params = {
                    proj: uniform_slerp(weights) for proj, weights in merged_params.items()
                }
            else:
                # 使用简单的平均加权合并
                merged_params = {
                    proj: torch.mean(torch.stack(weights), dim=0) for proj, weights in merged_params.items()
                }

            # 创建新专家
            new_expert = OlmoeMLP(model.config).to(dev)
            new_expert.gate_proj.weight.data = merged_params['gate_proj']
            new_expert.up_proj.weight.data = merged_params['up_proj']
            new_expert.down_proj.weight.data = merged_params['down_proj']
            new_experts.append(new_expert)

        # 更新路由层参数
        if new_experts:
            layer.mlp.experts = nn.ModuleList(new_experts)
            layer.mlp.num_experts = len(new_experts)
            # 更新门控权重
            new_gate_weight = torch.zeros((len(new_experts), gate.weight.shape[1]), device=dev)
            for new_idx, group in enumerate(expert_mapping):
                group_weights = torch.stack([gate.weight.data[old_idx].to(dev) for old_idx in group])
                new_gate_weight[new_idx] = group_weights.mean(dim=0)

            gate.weight = nn.Parameter(new_gate_weight)
            gate.out_features = len(new_experts)
            layer.mlp.top_k = min(layer.mlp.top_k, len(new_experts))

    return model

@torch.no_grad()
def model_expert_merging_noleading(model_name, all_original_expert_indices, model, expert_importance, expert_similarity, metrics, num_expert_group, dev):
    model.eval()

    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    # 计算各层的分组数
    num_layers = len(layers)
    layer_groups = [num_expert_group // num_layers] * num_layers
    for i in range(num_layers - 1, num_layers - (num_expert_group % num_layers) - 1, -1):
        layer_groups[i] += 1

    print("各层聚类分组配置:", layer_groups)
    
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        layer_key = f'layers.{layer_idx}.mlp'
        original_indices = all_original_expert_indices.get(layer_key, None)
        experts = layer.mlp.experts
        gate = layer.mlp.gate

        # 获取当前层的相似度矩阵（已预先计算）
        similarity_matrix = np.array(
            expert_similarity.get(f'model.layers.{layer_idx}.mlp', {}).get(metrics, None)
        )
        
        # 根据原始索引调整矩阵
        if original_indices is not None:
            similarity_matrix = similarity_matrix[np.ix_(original_indices, original_indices)]
        
        n_experts = similarity_matrix.shape[0]
        required_clusters = layer_groups[layer_idx]

        # 异常处理：当专家数不足时
        if n_experts < required_clusters:
            raise ValueError(f"Layer {layer_idx} 专家数({n_experts})少于分组数({required_clusters})")

        # 使用谱聚类（Spectral Clustering）进行分组
        from sklearn.cluster import SpectralClustering
        
        # 将相似度转换为亲和度矩阵（需要非负）
        affinity_matrix = (similarity_matrix + 1) / 2  # 转换到[0,1]范围
        
        cluster = SpectralClustering(
            n_clusters=required_clusters,
            affinity='precomputed',
            random_state=42,
            assign_labels='discretize'
        )
        
        clusters = cluster.fit_predict(affinity_matrix)

        # 检查聚类结果是否包含所有簇
        unique_clusters = np.unique(clusters)
        if len(unique_clusters) < required_clusters:
            raise ValueError(f"Layer {layer_idx} 聚类结果仅包含 {len(unique_clusters)} 个簇，但需要 {required_clusters} 个。")

        # 构建分组映射
        expert_mapping = [[] for _ in range(required_clusters)]
        for expert_idx, cluster_id in enumerate(clusters):
            expert_mapping[cluster_id].append(expert_idx)

        # 检查每个组是否有专家
        for cluster_id, group in enumerate(expert_mapping):
            if not group:
                raise ValueError(f"Layer {layer_idx} 的簇 {cluster_id} 中没有专家。请检查相似度矩阵或调整聚类参数。")

        # 修改参数合并部分
        new_experts = []
        for group in expert_mapping:
            # 初始化合并参数（显式创建零张量）
            merged_params = {
                'gate_proj': None,
                'up_proj': None,
                'down_proj': None
            }
            
            # 使用第一个专家参数初始化合并参数
            first_expert = experts[group[0]].to(dev)
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                merged_params[proj] = getattr(first_expert, proj).weight.data.clone()
            
            # 累加其他专家参数
            for expert_idx in group[1:]:
                expert = experts[expert_idx].to(dev)
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    w = getattr(expert, proj).weight.data
                    merged_params[proj] += w
            
            # 执行平均操作
            for proj in merged_params:
                merged_params[proj] /= len(group)
            
            # 创建新专家
            new_expert = OlmoeMLP(model.config).to(dev)
            new_expert.gate_proj.weight.data = merged_params['gate_proj']
            new_expert.up_proj.weight.data = merged_params['up_proj']
            new_expert.down_proj.weight.data = merged_params['down_proj']
            new_experts.append(new_expert)

        # 更新路由层
        layer.mlp.experts = nn.ModuleList(new_experts)
        layer.mlp.num_experts = len(new_experts)
        
        # 更新门控权重
        new_gate_weight = torch.zeros((len(new_experts), gate.weight.shape[1]), device=dev)
        for new_idx, group in enumerate(expert_mapping):
            group_weights = torch.stack([gate.weight.data[old_idx].to(dev) for old_idx in group])
            new_gate_weight[new_idx] = group_weights.mean(dim=0)
        
        gate.weight = nn.Parameter(new_gate_weight)
        gate.out_features = len(new_experts)
        layer.mlp.top_k = min(layer.mlp.top_k, len(new_experts))



@torch.no_grad()
def model_expert_SVD(model_name, model, profiling_mat, all_original_expert_indices, svd_method, ratio, mlp_lora_rank, dev):
    model.eval()        # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition...")
    for i in range(len(layers)):
        layer = layers[i]

        #### Replace MLP ####
        if "OLMoE" in model_name:
            svd_mlp = OlmoeSparseMoeBlock(config=model.config, ratio=ratio, rank=mlp_lora_rank).to(dev)
            svd_mlp.gate.weight.data = layer.mlp.gate.weight.data.to(dev)

        elif "Qwen" in model_name:
            # 未完成
            svd_mlp = OlmoeSparseMoeBlock(config=model.config, ratio=ratio)
        elif 'deepseekMoE' in model_name:
            # 未完成
            svd_mlp = OlmoeSparseMoeBlock(config=model.config, ratio=ratio)

        #### Replace MLP ####
        for n in range(len(layer.mlp.experts)):
            subset = find_layers(layer.mlp.experts[n])
            expert_original_index = all_original_expert_indices[f'layers.{i}.mlp'][n]
            for name in subset:
                W = subset[name].weight.data.float().to(dev)    # 从子层中提取权重矩阵W并转换为浮点类型
                svd_u, svd_v = multi_svd_implementaions(W, svd_method, mlp_lora_rank, i, name, profiling_mat, dev, expert_original_index)

                #### Replace MLP ####
                if 'OLMoE' in model_name:   
                    if "gate_proj" in name:
                        svd_mlp.experts[n].gate_u_proj.weight.data = svd_u.to(dev)
                        svd_mlp.experts[n].gate_v_proj.weight.data = svd_v.to(dev)
                    elif "down_proj" in name:
                        svd_mlp.experts[n].down_u_proj.weight.data = svd_u.to(dev)
                        svd_mlp.experts[n].down_v_proj.weight.data = svd_v.to(dev)
                    elif "up_proj" in name:
                        svd_mlp.experts[n].up_u_proj.weight.data = svd_u.to(dev)
                        svd_mlp.experts[n].up_v_proj.weight.data = svd_v.to(dev)
            # 清理不再使用的变量
            W = None
            del  W
        layer.mlp = svd_mlp
        del layer   # 删除当前层的引用
        torch.cuda.empty_cache()
    #print(model)
@torch.no_grad()

def model_expert_only_SVD(model_name, model, svd_method, ratio, mlp_lora_rank, dev, profiling_mat=None):
    model.eval()        # 将模型设置为评估模式，确保不启用训练时特有的操作如dropout等
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("mlp lora rank", mlp_lora_rank)
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

def evaluate_model_expert_importance(model_name, model, tokenizer, calib_loader, device):
    if model_name == 'OLMoE':
        return evaluate_olmoe_expert_importance_low_source(model_name, model, tokenizer, calib_loader, device)
    elif model_name == 'Qwen':
        return evaluate_qwen_expert_importance_low_source(model_name, model, tokenizer, calib_loader, device)

from tqdm import tqdm
import torch
from collections import defaultdict

# def evaluate_olmoe_expert_importance_low_source(model_name, model, tokenizer, calib_loader, device="cuda"):
#     """评估MoE模型专家重要性的完整流程
    
#     Args:
#         model: 原始模型对象
#         tokenizer: 对应分词器
#         calib_loader: 校准数据加载器 (返回文本列表)
#         device: 计算设备
        
#     Returns:
#         dict: 包含各层专家统计指标的嵌套字典
#     """
#     # 初始化包装器
#     model.to(device)
#     wrapped_model = OlmoeImportanceWrapper(model, device)
#     wrapped_model.eval()
    
#     # 显式重置统计指标
#     if hasattr(wrapped_model, 'reset_stats'):
#         wrapped_model.reset_stats()
#     else:
#         # 手动清空统计容器
#         for metric in wrapped_model.expert_stats.values():
#             metric.clear()

#     for batch in tqdm(calib_loader, desc="Calculating expert stats"):
#         torch.cuda.empty_cache()
        
#         # 保持原始数据格式处理逻辑
#         if isinstance(batch, dict):
#             batch = [batch]  # 统一转为字典列表格式
            
#         try:
#             # 使用您原有的预处理函数
#             inputs = prepare_inputs_for_model(batch, tokenizer, device)
#             # 确保数据在目标设备 (原始函数已包含to(device), 这里二次确认)
#             inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            
#             # 前向传播
#             _ = wrapped_model(**inputs)
            
#         except Exception as e:
#             print(f"处理批次时出错: {str(e)}")
#             print(f"批次类型: {type(batch)} 内容样例: {batch[0].keys() if isinstance(batch, list) else batch.keys()}")
#             raise

#         # 内存清理
#         del inputs
#         torch.cuda.empty_cache()

#     # 获取统计结果 (自动在CPU上)
#     stats = wrapped_model.get_full_stats()
    
#      # 后处理：转换numpy类型为Python原生类型
#     final_stats = {
#         'weights_sum': {},
#         'input_l2': {},
#         'input_count': {},
#         'output_l2': {},
#         'contribution': {}
#     }
#     for layer in stats['weights_sum'].keys():
#         final_stats['weights_sum'][layer] = [round(float(x), 4) for x in stats['weights_sum'][layer]]
#         final_stats['input_l2'][layer] = [round(float(x), 2) for x in stats['input_l2_sum'][layer]]
#         final_stats['input_count'][layer] = [int(x) for x in stats['input_count'][layer]]
#         final_stats['output_l2'][layer] = [round(float(x), 2) for x in stats['output_l2_sum'][layer]]
#         final_stats['contribution'][layer] = [round(float(x), 4) for x in stats['contribution'][layer]]

#     # 资源清理
#     del wrapped_model
#     torch.cuda.empty_cache()
    
#     return final_stats

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


def plot_expert_similarity(router_logits, expert_similarity):
    """
    可视化专家相似性和 router logits。
    
    参数:
        router_logits: 每一层的 router logits。
        expert_similarity: 每一层的专家相似性矩阵。
    """
    import matplotlib.pyplot as plt
    output_dir = "fig_outfile/router_logits"
    os.makedirs(output_dir, exist_ok=True)

    for layer, sim_matrix in expert_similarity.items():
        plt.figure(figsize=(10, 8))
        plt.imshow(sim_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Expert Similarity Matrix for Layer {layer}')
        plt.xlabel('Experts')
        plt.ylabel('Experts')

        filename = os.path.join(output_dir, f"{layer}_Expert_Similarity_Matrix.png")
        plt.savefig(filename)
        plt.close()

        # 可视化 router logits
        logits = router_logits[layer]
        logits_mean = np.mean(logits, axis=0)  # 计算每个专家的平均 logits
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(logits_mean)), logits_mean)
        plt.title(f'Concatenated Router Logits for Layer {layer}')
        plt.xlabel('Experts')
        plt.ylabel('Average Logits')
        filename = os.path.join(output_dir, f"{layer}_Router_Logits.png")
        plt.savefig(filename)


def model_original_ppl(model, tokenizer, dev):
    # 定义一个简单的输入句子
    input_sentence = "Hello, who design you"

    # 使用 tokenizer 处理输入
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(dev)

    # 获取模型的输出
    with torch.no_grad():  # 确保不计算梯度以节省内存和时间
        outputs = model(**inputs)

    # 打印 logits 或其他您感兴趣的输出部分
    print("Model output logits:", outputs.logits)
    
    # 如果是生成任务，可以解码生成的token ids为文本
    if hasattr(outputs, 'logits'):
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        print("Decoded model output:", decoded_text)
