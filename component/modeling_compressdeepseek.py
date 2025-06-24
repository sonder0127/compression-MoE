from torch import nn
from transformers.activations import ACT2FN
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.utils.checkpoint
#from transformers.models.deepseek.modeling_deepseek import DeepseekMoE
from transformers.activations import ACT2FN
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
#========================================== SVD =======================================================
class SVD_DeepseekMLP(nn.Module):
    def __init__(self, config, ratio=1, lora_rank=602, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.lora_rank = lora_rank
        
        # LoRA modifications
        self.gate_v_proj = nn.Linear(self.hidden_size, lora_rank, bias=False)
        self.gate_u_proj = nn.Linear(lora_rank, self.intermediate_size, bias=False)
        
        self.up_v_proj = nn.Linear(self.hidden_size, lora_rank, bias=False)
        self.up_u_proj = nn.Linear(lora_rank, self.intermediate_size, bias=False)
        
        self.down_v_proj = nn.Linear(self.intermediate_size, lora_rank, bias=False)
        self.down_u_proj = nn.Linear(lora_rank, self.hidden_size, bias=False)
        
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            
            # For LoRA versions, you need to handle the slicing logic accordingly.
            gate_v_slices = self.gate_v_proj.weight.split(slice, dim=0)
            up_v_slices = self.up_v_proj.weight.split(slice, dim=0)
            down_v_slices = self.down_v_proj.weight.split(slice, dim=1)

            gate_proj_slices = [self.gate_u_proj(F.linear(x, gate_v_slices[i])) for i in range(self.config.pretraining_tp)]
            up_proj_slices = [self.up_u_proj(F.linear(x, up_v_slices[i])) for i in range(self.config.pretraining_tp)]

            gate_proj = torch.cat(gate_proj_slices, dim=-1)
            up_proj = torch.cat(up_proj_slices, dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj_slices = [
                F.linear(intermediate_states[i], down_v_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum([self.down_u_proj(down_proj_slice) for down_proj_slice in down_proj_slices])
        else:
            gate_proj = self.gate_u_proj(self.gate_v_proj(x))
            up_proj = self.up_u_proj(self.up_v_proj(x))

            intermediate_output = self.act_fn(gate_proj) * up_proj
            down_proj_slices = self.down_v_proj(intermediate_output).split(self.lora_rank, dim=-1)
            down_proj = sum([self.down_u_proj(down_proj_slice) for down_proj_slice in down_proj_slices])

        return down_proj

# ========================================== Pruning ===================================================
# class DeepseekMoeImportanceWrapper(torch.nn.Module):
#     def __init__(self, model, device):
#         super().__init__()
#         self.model = model.to(device)
#         self.device = device
#         self.expert_counts = {}          # 专家激活次数统计
#         self.expert_weights_sum = {}     # 路由权重累加值
#         self.expert_input_l2_sum = {}    # 输入L2范数总和
#         self.expert_input_count = {}     # 输入样本计数
#         self.expert_output_l2_sum = {}   # 输出L2范数总和
#         self.hooks = []
#         self._register_hooks()

#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)

#     def _register_hooks(self):
#         """注册钩子适配Deepseek-MoE结构"""
#         for name, module in self.model.named_modules():
#             if isinstance(module, DeepseekMoE):
#                 # 初始化普通专家统计
#                 num_regular = module.config.n_routed_experts
#                 self.expert_counts[name] = torch.zeros(num_regular, device=self.device)
#                 self.expert_weights_sum[name] = torch.zeros(num_regular, device=self.device)
#                 self.expert_input_l2_sum[name] = torch.zeros(num_regular, device=self.device)
#                 self.expert_input_count[name] = torch.zeros(num_regular, device=self.device)
#                 self.expert_output_l2_sum[name] = torch.zeros(num_regular, device=self.device)
                
#                 # 注册MoE层路由钩子
#                 moe_hook = module.register_forward_hook(self._moe_layer_hook(name))
#                 self.hooks.append(moe_hook)
                
#                 # 注册普通专家前向钩子
#                 for expert_idx in range(num_regular):
#                     expert = module.experts[expert_idx]
#                     expert_hook = expert.register_forward_hook(
#                         self._expert_forward_hook(name, expert_idx)
#                     )
#                     self.hooks.append(expert_hook)
                
#                 # 注册共享专家钩子（如果存在）
#                 if hasattr(module, 'shared_experts') and module.shared_experts is not None:
#                     shared_hook = module.shared_experts.register_forward_hook(
#                         self._shared_expert_hook(name)
#                     )
#                     self.hooks.append(shared_hook)

#     def _moe_layer_hook(self, layer_name):
#         """路由统计逻辑（适配Deepseek的路由机制）"""
#         def hook(module, inputs, outputs):
#             # 通过gate重新计算路由信息
#             hidden_states = inputs[0]
#             with torch.no_grad():
#                 topk_idx, topk_weight, _ = module.gate(hidden_states)
            
#             # 展平专家选择索引
#             selected_experts = topk_idx.view(-1)
#             # 统计激活次数
#             unique, counts = torch.unique(selected_experts, return_counts=True)
#             self.expert_counts[layer_name][unique] += counts.to(self.device)
            
#             # 累加路由权重（按专家索引分组）
#             topk_weight_flat = topk_weight.view(-1)
#             for expert_idx in unique:
#                 mask = (selected_experts == expert_idx)
#                 self.expert_weights_sum[layer_name][expert_idx] += topk_weight_flat[mask].sum().to(self.device)
        
#         return hook

#     def _expert_forward_hook(self, layer_name, expert_idx):
#         """普通专家统计（输入输出特征分析）"""
#         def hook(module, inputs, outputs):
#             # 提取并转换输入输出
#             input_tensor = inputs[0].detach().to(self.device)
#             output_tensor = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs[0].detach().to(self.device)
            
#             # 计算L2范数均值
#             input_l2 = torch.norm(input_tensor, p=2, dim=-1).mean()
#             output_l2 = torch.norm(output_tensor, p=2, dim=-1).mean()
            
#             # 安全更新统计量
#             if not torch.isnan(input_l2) and not torch.isinf(input_l2):
#                 self.expert_input_l2_sum[layer_name][expert_idx] += input_l2
#             if not torch.isnan(output_l2) and not torch.isinf(output_l2):
#                 self.expert_output_l2_sum[layer_name][expert_idx] += output_l2
#             self.expert_input_count[layer_name][expert_idx] += 1
        
#         return hook

#     def _shared_expert_hook(self, layer_name):
#         """共享专家独立统计"""
#         def hook(module, inputs, outputs):
#             input_tensor = inputs[0].detach().to(self.device)
#             output_tensor = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs[0].detach().to(self.device)
            
#             input_l2 = torch.norm(input_tensor, p=2, dim=-1).mean()
#             output_l2 = torch.norm(output_tensor, p=2, dim=-1).mean()
            
#             key = f"{layer_name}_shared"
#             if not torch.isnan(input_l2) and not torch.isinf(input_l2):
#                 self.expert_input_l2_sum[key] = self.expert_input_l2_sum.get(key, 0) + input_l2
#             if not torch.isnan(output_l2) and not torch.isinf(output_l2):
#                 self.expert_output_l2_sum[key] = self.expert_output_l2_sum.get(key, 0) + output_l2
#             self.expert_input_count[key] = self.expert_input_count.get(key, 0) + 1
        
#         return hook

#     def reset_counts(self):
#         """重置所有统计计数器"""
#         for stats_dict in [self.expert_counts, self.expert_weights_sum,
#                           self.expert_input_l2_sum, self.expert_output_l2_sum,
#                           self.expert_input_count]:
#             for k in stats_dict:
#                 if isinstance(stats_dict[k], torch.Tensor):
#                     stats_dict[k].zero_()
#                 else:
#                     stats_dict[k] = 0

#     def get_expert_stats(self):
#         """生成标准化统计结果"""
#         stats = {
#             'activation_frequency': {},
#             'sum_routing_weights': {},
#             'input_l2_avg': {},
#             'output_l2_avg': {},
#             'shared_expert_stats': {}
#         }
        
#         # 处理普通专家数据
#         for layer in self.expert_counts:
#             # 计算激活频率
#             total_activations = self.expert_counts[layer].sum().item()
#             stats['activation_frequency'][layer] = (
#                 (self.expert_counts[layer] / total_activations).tolist()
#                 if total_activations > 0 else
#                 [0.0] * len(self.expert_counts[layer])
#             )
            
#             # 路由权重总和
#             stats['sum_routing_weights'][layer] = self.expert_weights_sum[layer].tolist()
            
#             # 计算平均L2范数
#             input_avgs = []
#             output_avgs = []
#             for idx in range(len(self.expert_counts[layer])):
#                 count = self.expert_input_count[layer][idx].item()
#                 input_avgs.append(
#                     (self.expert_input_l2_sum[layer][idx] / count).item()
#                     if count > 0 else 0.0
#                 )
#                 output_avgs.append(
#                     (self.expert_output_l2_sum[layer][idx] / count).item()
#                     if count > 0 else 0.0
#                 )
#             stats['input_l2_avg'][layer] = input_avgs
#             stats['output_l2_avg'][layer] = output_avgs
        
#         # 处理共享专家数据
#         shared_stats = {}
#         for key in list(self.expert_input_count.keys()):
#             if "_shared" in key:
#                 base_layer = key.replace("_shared", "")
#                 count = self.expert_input_count[key]
#                 shared_stats[base_layer] = {
#                     'shared_input_l2': (
#                         self.expert_input_l2_sum[key] / count if count > 0 else 0.0
#                     ).item(),
#                     'shared_output_l2': (
#                         self.expert_output_l2_sum[key] / count if count > 0 else 0.0
#                     ).item()
#                 }
#         stats['shared_expert_stats'] = shared_stats
        
#         return stats

#     def remove_hooks(self):
#         """安全移除所有注册的钩子"""
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks.clear()