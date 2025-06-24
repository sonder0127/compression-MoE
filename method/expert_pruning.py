import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
from ..component.modeling_pruning import PrunableOlmoeSparseMoeBlockWrapper
from ..component.modeling_compressqwen import PrunableQwenMoeSparseMoeBlockWrapper
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM
logger = logging.getLogger(__name__)

def model_expert_pruning_easy(dev, model: OlmoeForCausalLM, calib_loader: DataLoader, experts_importance, 
                         strategy='threshold', threshold=0.05, num_experts_remain=None):
    """
    对 Olmoe 模型的每一层进行专家剪枝，支持两种剪枝策略：
    1. 'threshold'：剪去重要性低于阈值 threshold 的专家；
    2. 'fixed'：每层保留固定数量的专家（根据专家重要性排序，保留重要性较高的 num_experts_remain 个专家）。
    
    同时记录下每一层剪枝后保留的专家的原始序号，存储在 all_original_expert_indices 中，并一并返回。
    """

    #计算出每层专家剩余的专家个数保存到一个列表中，然后再

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

        # 根据选择的剪枝策略进行剪枝
        if strategy == 'threshold':
            b.prune_by_threshold(current_layer_experts_importance, threshold)
        elif strategy == 'fixed':
            if num_experts_remain is None:
                raise ValueError("For fixed strategy, num_experts_remain must be provided.")
            b.prune_by_importance(current_layer_experts_importance, num_experts_remain)
        else:
            raise ValueError("Unknown pruning strategy. Use 'threshold' or 'fixed'.")

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
    print('============', model.num_experts)

    # 返回模型、global_loss_history 以及所有层剪枝后保留的原始专家索引
    return model, global_loss_history, all_original_expert_indices

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
    print('============', model.num_experts)

    # 返回模型、global_loss_history 以及所有层剪枝后保留的原始专家索引
    return model, all_original_expert_indices

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
