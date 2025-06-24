from torch import nn
from transformers.activations import ACT2FN
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
import logging
logger = logging.getLogger(__name__)
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock
from typing import Optional

# ====================================== SVD ========================================
class SVD_Qwen2MoeMLP(nn.Module):
    def __init__(self, config, intermediate_size=None, ratio=1, lora_rank=602):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.lora_rank = lora_rank

        self.gate_u_proj = nn.Linear(lora_rank, self.intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(self.hidden_size, lora_rank, bias=False)
        
        self.down_u_proj = nn.Linear(lora_rank, self.hidden_size, bias=False)
        self.down_v_proj = nn.Linear(self.intermediate_size, lora_rank, bias=False)
        
        self.up_u_proj = nn.Linear(lora_rank, self.intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(self.hidden_size, lora_rank, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        up = self.up_u_proj(self.up_v_proj(x))
        gate = self.gate_u_proj(self.gate_v_proj(x))
        return self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))

# ====================================== Pruning ========================================

# importance save
class QwenMoeImportanceWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.expert_counts = {}          # 激活次数统计（与OLMoE完全一致）
        self.expert_weights_sum = {}     # 路由权重累加值
        self.expert_input_l2_sum = {}    # 输入L2范数总和 
        self.expert_input_count = {}     # 输入样本计数
        self.expert_output_l2_sum = {}   # 输出L2范数总和
        self.hooks = []
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _register_hooks(self):
        """注册钩子的逻辑适配Qwen结构"""
        for name, module in self.model.named_modules():
            if isinstance(module, Qwen2MoeSparseMoeBlock):
                # 初始化统计存储（普通专家 + 共享专家）
                num_regular = module.num_experts
                self.expert_counts[name] = torch.zeros(num_regular, device=self.device)
                self.expert_weights_sum[name] = torch.zeros(num_regular, device=self.device)
                self.expert_input_l2_sum[name] = torch.zeros(num_regular, device=self.device)
                self.expert_input_count[name] = torch.zeros(num_regular, device=self.device)
                self.expert_output_l2_sum[name] = torch.zeros(num_regular, device=self.device)
                
                # 注册MoE层钩子
                moe_hook = module.register_forward_hook(self._moe_layer_hook(name))
                self.hooks.append(moe_hook)
                
                # 注册普通专家钩子
                for expert_idx in range(num_regular):
                    expert = module.experts[expert_idx]
                    expert_hook = expert.register_forward_hook(
                        self._expert_forward_hook(name, expert_idx)
                    )
                    self.hooks.append(expert_hook)
                
                # 注册共享专家钩子（单独统计）
                shared_hook = module.shared_expert.register_forward_hook(
                    self._shared_expert_hook(name)
                )
                self.hooks.append(shared_hook)

    def _moe_layer_hook(self, layer_name):
        """路由统计钩子（适配Qwen的路由逻辑）"""
        def hook(module, inputs, outputs):
            final_hidden, router_logits = outputs  # Qwen输出格式
            
            # 获取路由权重（仅普通专家）
            routing_weights = F.softmax(router_logits, dim=-1)
            top_k = min(module.top_k, module.num_experts)
            top_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
            
            # 更新统计量（与OLMoE逻辑一致）
            unique, counts = torch.unique(selected_experts, return_counts=True)
            self.expert_counts[layer_name][unique] += counts.to(self.device)
            
            # 累加路由权重（保持与OLMoE相同的计算方式）
            for expert_idx in unique:
                mask = (selected_experts == expert_idx)
                self.expert_weights_sum[layer_name][expert_idx] += top_weights[mask].sum().to(self.device)
                
        return hook

    def _expert_forward_hook(self, layer_name, expert_idx):
        """普通专家钩子（保持与OLMoE相同的统计逻辑）"""
        def hook(module, inputs, outputs):
            # 输入输出统计（适配Qwen的专家结构）
            input_tensor = inputs[0].detach().to(self.device)
            output_tensor = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs[0].detach().to(self.device)
            
            # 计算L2范数（与OLMoE完全一致）
            input_l2 = torch.norm(input_tensor, p=2, dim=-1).mean()
            output_l2 = torch.norm(output_tensor, p=2, dim=-1).mean()
            
            # 安全更新（防止NaN/Inf）
            if not torch.isnan(input_l2) and not torch.isinf(input_l2):
                self.expert_input_l2_sum[layer_name][expert_idx] += input_l2
            if not torch.isnan(output_l2) and not torch.isinf(output_l2):
                self.expert_output_l2_sum[layer_name][expert_idx] += output_l2
            self.expert_input_count[layer_name][expert_idx] += 1
            
        return hook

    def _shared_expert_hook(self, layer_name):
        """共享专家钩子（独立统计，不计入常规指标）"""
        def hook(module, inputs, outputs):
            # 独立统计共享专家（不影响原有指标）
            input_tensor = inputs[0].detach().to(self.device)
            output_tensor = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs[0].detach().to(self.device)
            
            # 计算L2范数但单独存储
            input_l2 = torch.norm(input_tensor, p=2, dim=-1).mean()
            output_l2 = torch.norm(output_tensor, p=2, dim=-1).mean()
            
            key = f"{layer_name}_shared"
            if not torch.isnan(input_l2) and not torch.isinf(input_l2):
                self.expert_input_l2_sum[key] = self.expert_input_l2_sum.get(key, 0) + input_l2
            if not torch.isnan(output_l2) and not torch.isinf(output_l2):
                self.expert_output_l2_sum[key] = self.expert_output_l2_sum.get(key, 0) + output_l2
            self.expert_input_count[key] = self.expert_input_count.get(key, 0) + 1
            
        return hook

    def reset_counts(self):
        """重置统计（与OLMoE方法完全一致）"""
        for stats_dict in [self.expert_counts, self.expert_weights_sum, 
                          self.expert_input_l2_sum, self.expert_output_l2_sum,
                          self.expert_input_count]:
            for k in stats_dict:
                if isinstance(stats_dict[k], torch.Tensor):
                    stats_dict[k].zero_()
                else:
                    stats_dict[k] = 0

    def get_expert_stats(self):
        """生成统计结果（保持与OLMoE完全相同的输出格式）"""
        stats = {
            'activation_frequency': {},
            'sum_routing_weights': {},
            'input_l2_avg': {},
            'output_l2_avg': {}
        }
        
        # 处理普通专家
        for layer in self.expert_counts:
            # 激活频率
            total = self.expert_counts[layer].sum().item()
            stats['activation_frequency'][layer] = (
                (self.expert_counts[layer] / total).tolist() if total > 0 
                else [0.0] * len(self.expert_counts[layer])
            )
            
            # 路由权重累加值（直接取值）
            stats['sum_routing_weights'][layer] = self.expert_weights_sum[layer].tolist()
            
            # 输入输出L2平均值
            input_avg = []
            output_avg = []
            for idx in range(len(self.expert_counts[layer])):
                count = self.expert_input_count[layer][idx].item()
                input_avg.append(
                    (self.expert_input_l2_sum[layer][idx] / count).item() if count > 0 else 0.0
                )
                output_avg.append(
                    (self.expert_output_l2_sum[layer][idx] / count).item() if count > 0 else 0.0
                )
            stats['input_l2_avg'][layer] = input_avg
            stats['output_l2_avg'][layer] = output_avg
            
        # 添加共享专家信息（独立字段）
        shared_stats = {}
        for key in self.expert_input_count:
            if "_shared" in key:
                base_layer = key.replace("_shared", "")
                count = self.expert_input_count[key]
                shared_stats[base_layer] = {
                    'shared_input_l2': (
                        self.expert_input_l2_sum[key] / count if count > 0 else 0.0
                    ).item(),
                    'shared_output_l2': (
                        self.expert_output_l2_sum[key] / count if count > 0 else 0.0
                    ).item()
                }
        stats['shared_expert_stats'] = shared_stats
        
        return stats

    def remove_hooks(self):
        """移除钩子（与OLMoE方法一致）"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


# pruning
class CacheDataset:
    def __init__(self):
        self.Xs = []
        self.Zs = []

    def append(self, X=None, Z=None):
        if X is not None:
            self.Xs.append(X)
        if Z is not None:
            self.Zs.append(Z)
            
class PrunableQwenMoeSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model, r: Optional[int] = None):
        super().__init__()
        if isinstance(model, Qwen2MoeSparseMoeBlock):
            self.model = model
        else:
            # 若模型被包裹在其他结构中，需调整访问路径
            self.model = model.model  
        
        self.r = r
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False
        self.original_expert_indices = None  # 保留原始专家索引

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)

        # 应用专家剪枝掩码
        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k = min(self.model.top_k, routing_weights.size(-1))
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        
        if self.model.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # 普通专家计算
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), 
            dtype=hidden_states.dtype, 
            device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.model.num_experts):
            if expert_idx in (self.experts_to_drop or []):
                continue
            
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        # 共享专家计算（保持不变）
        shared_expert_output = self.model.shared_expert(hidden_states)
        shared_gate = F.sigmoid(self.model.shared_expert_gate(hidden_states))
        final_hidden_states = final_hidden_states + shared_gate * shared_expert_output

        # 缓存逻辑
        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warn(f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(
            X=(hidden_states if self.cache_X else None),
            Z=(final_hidden_states if self.cache_Z else None)
        )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

    @torch.no_grad()
    def prune_by_threshold(self, experts_importance, threshold):
        """基于重要性阈值剪枝"""
        # 只处理普通专家，保留共享专家
        valid_experts = [i for i in range(len(experts_importance)) if i not in getattr(self.model, 'shared_expert_idx', [])]
        filtered_importance = [experts_importance[i] for i in valid_experts]
        
        # 确定要剪枝的专家
        self.experts_to_drop = [
            valid_experts[i] 
            for i, imp in enumerate(filtered_importance) 
            if imp < threshold
        ]
        experts_to_keep = [i for i in valid_experts if i not in self.experts_to_drop]

        # 异常处理
        if len(experts_to_keep) == 0:
            logger.warning("No experts meet the threshold. Keeping top-1 expert.")
            experts_to_keep = [np.argmax(filtered_importance).item()]

        # 更新门控层和专家列表
        self._update_gate_and_experts(experts_to_keep)
        self.original_expert_indices = experts_to_keep

    @torch.no_grad()
    def prune_by_importance(self, experts_importance, num_experts_remain):
        """基于重要性排名剪枝"""
        valid_experts = [i for i in range(len(experts_importance)) if i not in getattr(self.model, 'shared_expert_idx', [])]
        filtered_importance = [experts_importance[i] for i in valid_experts]
        
        # 选择top-k专家
        sorted_indices = np.argsort(filtered_importance)[-num_experts_remain:]
        experts_to_keep = [valid_experts[i] for i in sorted_indices]
        self.experts_to_drop = list(set(range(self.model.num_experts)) - set(experts_to_keep))

        self._update_gate_and_experts(experts_to_keep)
        self.original_expert_indices = experts_to_keep

    def _update_gate_and_experts(self, experts_to_keep):
        """更新门控层和专家列表的公共方法"""
        # 创建新门控层
        new_gate = nn.Linear(
            in_features=self.model.gate.in_features,
            out_features=len(experts_to_keep),
            bias=False,
            device=self.model.gate.weight.device,
            dtype=self.model.gate.weight.dtype
        )
        new_gate.weight.data.copy_(self.model.gate.weight.data[experts_to_keep])
        self.model.gate = new_gate

        # 更新专家列表
        self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
        self.model.num_experts = len(experts_to_keep)
        self.model.top_k = min(self.model.top_k, self.model.num_experts)  # 动态调整top_k

    def reset_pruning(self):
        """重置剪枝状态"""
        self.experts_to_drop = None
        if self.original_expert_indices is not None:
            self._update_gate_and_experts(self.original_expert_indices)