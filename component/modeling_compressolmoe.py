from torch import nn
from transformers.activations import ACT2FN
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.utils.checkpoint
import logging
logger = logging.getLogger(__name__)
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
from typing import Optional

# ====================================== SVD ========================================
class SVD_OlmoeMLP(nn.Module):
    def __init__(self, config, ratio=1, lora_rank=602):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
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


#===================================Pruning========================================

# olmoe pruning importance 
class OlmoeImportanceWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super(OlmoeImportanceWrapper, self).__init__()
        self.model = model.to(device)  # 确保模型也在device上
        self.device = device
        self.expert_counts = {}          # 激活次数统计
        self.expert_weights_sum = {}     # 路由权重累加值
        self.expert_input_l2_sum = {}    # 输入L2范数总和
        self.expert_input_count = {}     # 输入样本计数
        self.expert_output_l2_sum = {}   # 输出L2范数总和
        self.hooks = []
        self._register_hooks()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):  # 替换成实际的MoE层类型
                num_experts = module.num_experts
                self.expert_counts[name] = torch.zeros(num_experts, device=self.device)
                self.expert_weights_sum[name] = torch.zeros(num_experts, device=self.device)
                self.expert_input_l2_sum[name] = torch.zeros(num_experts, device=self.device)
                self.expert_input_count[name] = torch.zeros(num_experts, device=self.device)
                self.expert_output_l2_sum[name] = torch.zeros(num_experts, device=self.device)
                
                moe_hook = module.register_forward_hook(self._moe_layer_hook(name))
                self.hooks.append(moe_hook)
                
                for expert_idx, expert in enumerate(module.experts):
                    expert_hook = expert.register_forward_hook(
                        self._expert_forward_hook(name, expert_idx))
                    self.hooks.append(expert_hook)

    def _moe_layer_hook(self, layer_name):
        def hook(module, inputs, outputs):
            final_hidden_states, router_logits = outputs
            router_weights = torch.softmax(router_logits, dim=-1).to(self.device)
            selected_experts = torch.argmax(router_weights, dim=-1)
            
            unique, counts = torch.unique(selected_experts, return_counts=True)
            self.expert_counts[layer_name][unique] += counts
            
            selected_weights = router_weights[torch.arange(router_weights.size(0)), selected_experts]
            for expert_idx in unique:
                mask = (selected_experts == expert_idx)
                self.expert_weights_sum[layer_name][expert_idx] += selected_weights[mask].sum()
                
        return hook

    def _expert_forward_hook(self, layer_name, expert_idx):
        def hook(module, inputs, outputs):
            input_tensor = inputs[0].detach().to(self.device)
            output_tensor = outputs.detach() if isinstance(outputs, torch.Tensor) else outputs[0].detach().to(self.device)
            
            input_l2 = torch.norm(input_tensor, p=2, dim=-1).mean()
            output_l2 = torch.norm(output_tensor, p=2, dim=-1).mean()
            
            # 防止NaN
            if not torch.isnan(input_l2) and not torch.isinf(input_l2):
                self.expert_input_l2_sum[layer_name][expert_idx] += input_l2
            if not torch.isnan(output_l2) and not torch.isinf(output_l2):
                self.expert_output_l2_sum[layer_name][expert_idx] += output_l2
            self.expert_input_count[layer_name][expert_idx] += 1
            
        return hook

    def reset_counts(self):
        for stats_dict in [self.expert_counts, self.expert_weights_sum, 
                           self.expert_input_l2_sum, self.expert_output_l2_sum, 
                           self.expert_input_count]:
            for tensor in stats_dict.values():
                tensor.zero_()

    def get_expert_stats(self):
        stats = {
            'activation_frequency': {},
            'sum_routing_weights': {},
            'input_l2_avg': {},
            'output_l2_avg': {}
        }
        
        for layer in self.expert_counts:
            total_activations = self.expert_counts[layer].sum()
            if total_activations > 0:
                stats['activation_frequency'][layer] = (
                    (self.expert_counts[layer] / total_activations).tolist()
                )
            else:
                stats['activation_frequency'][layer] = [0] * len(self.expert_counts[layer])
            
            stats['sum_routing_weights'][layer] = self.expert_weights_sum[layer].tolist()
            
            input_l2_avg = []
            output_l2_avg = []
            for idx in range(len(self.expert_counts[layer])):
                count = self.expert_input_count[layer][idx]
                if count > 0:
                    avg_input_l2 = self.expert_input_l2_sum[layer][idx] / count
                    avg_output_l2 = self.expert_output_l2_sum[layer][idx] / count
                    # 防止NaN
                    if not torch.isnan(avg_input_l2) and not torch.isinf(avg_input_l2):
                        input_l2_avg.append(avg_input_l2.item())
                    else:
                        input_l2_avg.append(0.0)
                        
                    if not torch.isnan(avg_output_l2) and not torch.isinf(avg_output_l2):
                        output_l2_avg.append(avg_output_l2.item())
                    else:
                        output_l2_avg.append(0.0)
                else:
                    input_l2_avg.append(0.0)
                    output_l2_avg.append(0.0)
                    
            stats['input_l2_avg'][layer] = input_l2_avg
            stats['output_l2_avg'][layer] = output_l2_avg
            
        return stats

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

# olmoe pruning
class CacheDataset:
    def __init__(self):
        self.Xs = []
        self.Zs = []

    def append(self, X=None, Z=None):
        if X is not None:
            self.Xs.append(X)
        if Z is not None:
            self.Zs.append(Z)

class PrunableOlmoeSparseMoeBlockWrapper(nn.Module):
    def __init__(self, model, r: Optional[int] = None):
        super().__init__()
        if isinstance(model, OlmoeSparseMoeBlock):
            self.model = model
        else:
            self.model = model.model
        self.r = r
        
        self.experts_to_drop = None
        self.cache_space = CacheDataset()
        self.cache_logits = False
        self.cache_X = False
        self.cache_Z = False

        # 新增属性，用于记录剪枝后保留的原始专家序号
        self.original_expert_indices = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)

        print('========self.experts_to_drop', self.experts_to_drop)
        if self.experts_to_drop is not None:
            for e in self.experts_to_drop:
                router_logits[:, e] = -float('inf')

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        
        # Ensure top_k does not exceed the number of experts
        top_k = min(self.model.top_k, routing_weights.size(-1))
        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        if self.model.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        if self.experts_to_drop is not None and (self.cache_logits or self.cache_X or self.cache_Z):
            logger.warn(f'Already dropped {self.experts_to_drop} but still storing activations.')
        self.cache_space.append(X=(hidden_states if self.cache_X else None), 
                                Z=(final_hidden_states if self.cache_Z else None))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    @torch.no_grad()
    def prune_by_threshold(self, experts_importance, threshold):
        # 根据阈值剪枝：丢弃重要性低于阈值的专家
        self.experts_to_drop = [i for i, importance in enumerate(experts_importance) if importance < threshold]
        print('===self.experts_to_drop===', self.experts_to_drop)
        # 保留重要性高于等于阈值的专家
        experts_to_keep = [i for i, importance in enumerate(experts_importance) if importance >= threshold]
        
        if len(experts_to_keep) == 0:
            logger.warning("No experts meet the threshold. Keeping all experts.")
            experts_to_keep = list(range(len(experts_importance)))
        
        # 记录下原始保留的专家序号
        self.original_expert_indices = experts_to_keep
        
        gate_new = nn.Linear(in_features=self.model.gate.in_features,
                             out_features=len(experts_to_keep), bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(experts_to_keep)]
        self.model.gate = gate_new

        self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
        self.model.num_experts = len(experts_to_keep)

        # Adjust top_k if necessary
        self.model.top_k = min(self.model.top_k, self.model.num_experts)

    @torch.no_grad()
    def prune_by_importance(self, experts_importance, num_experts_remain):
        # 固定数量剪枝：根据专家重要性排序，保留 num_experts_remain 个专家
        num_experts_to_keep = num_experts_remain
        importance_scores = experts_importance.copy()
        
        # 按重要性从小到大排序，选出排名靠后的专家保留
        sorted_indices = sorted(range(len(importance_scores)), key=lambda k: importance_scores[k])
        experts_to_keep = sorted_indices[-num_experts_to_keep:]
        self.experts_to_drop = set(range(self.model.num_experts)) - set(experts_to_keep)
        
        # 记录下原始保留的专家序号
        self.original_expert_indices = experts_to_keep
        
        gate_new = nn.Linear(in_features=self.model.gate.in_features,
                             out_features=num_experts_remain, bias=False, device='cpu', dtype=torch.bfloat16)
        gate_new.weight.data = self.model.gate.weight.data[list(experts_to_keep)]
        print('===self.experts_to_drop===', self.experts_to_drop)
        self.model.gate = gate_new

        self.model.experts = nn.ModuleList([self.model.experts[i] for i in experts_to_keep])
        self.model.num_experts = num_experts_remain
        
#===================================Merging========================================





# ====================================== basenet ========================================
class SVD_OlmoeMLP_weight(nn.Module):
    def __init__(self, config, expert_group_index=0, ratio=1, rank=602):
        super().__init__()  # 继承 nn.Module 的初始化方法
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ratio = ratio
        self.lora_rank = rank
        low_rank = int(self.intermediate_size * self.hidden_size * self.ratio / (self.intermediate_size + self.hidden_size))
        self.expert_group_index = int(expert_group_index)
        
        # Initialize weights for low-rank projections
        self.layers = nn.ModuleDict({
            'gate': nn.Sequential(
                nn.Linear(self.hidden_size, self.lora_rank, bias=False),
                nn.Linear(self.lora_rank, self.intermediate_size, bias=False)
            ),
            'up': nn.Sequential(
                nn.Linear(self.hidden_size, self.lora_rank, bias=False),
                nn.Linear(self.lora_rank, self.intermediate_size, bias=False)
            ),
            'down': nn.Sequential(
                nn.Linear(self.intermediate_size, self.lora_rank, bias=False),
                nn.Linear(self.lora_rank, self.hidden_size, bias=False)
            )
        })

class SVD_Basenet_weight(nn.Module):
    def __init__(self, config):
        super().__init__()  # 继承 nn.Module 的初始化方法
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.hidden_act = config.hidden_act
        
        # Initialize parameters as nn.Parameter objects and store them in a dictionary
        self.layers = nn.ModuleDict({
            'gate': nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            'up': nn.Linear(self.hidden_size, self.intermediate_size, bias=False),
            'down': nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        })


class SVD_Basenet_OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config, num_group=1, ratio=1, rank=602):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.rank = rank
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        self.experts = nn.ModuleList([SVD_OlmoeMLP_weight(config, ratio=ratio, rank=rank) for _ in range(self.num_experts)])
        self.basenet = nn.ModuleList([SVD_Basenet_weight(config) for _ in range(num_group)])

    def basenet_forward(self, x, expert_index):
        #print(x.shape)   #[299,2048],[401.2048]
        basenet_index = self.experts[expert_index].expert_group_index

        gate_proj, up_proj, down_proj = self.basenet[basenet_index].layers['gate'], self.basenet[basenet_index].layers['up'], self.basenet[basenet_index].layers['down']
        #print(gate_proj.weight.shape, up_proj.weight.shape, down_proj.weight.shape)  #[1024, 2048]) torch.Size([1024, 2048]) torch.Size([2048, 1024]
        # Access the sub-modules within Sequential using indices
        gate_proj_lora_A, gate_proj_lora_B = self.experts[expert_index].layers['gate'][0], self.experts[expert_index].layers['gate'][1]
        up_proj_lora_A, up_proj_lora_B = self.experts[expert_index].layers['up'][0], self.experts[expert_index].layers['up'][1]
        down_proj_lora_A, down_proj_lora_B = self.experts[expert_index].layers['down'][0], self.experts[expert_index].layers['down'][1]

        act_fn = ACT2FN[self.basenet[basenet_index].config.hidden_act]

        gate_output = gate_proj(x)
        up_output = up_proj(x)
        #print(gate_output.shape) #299, 1024]) [401, 1024])
        gate_output_lora = gate_proj_lora_B(gate_proj_lora_A(x))
        up_output_lora = up_proj_lora_B(up_proj_lora_A(x))

        adjusted_gate_output = gate_output + gate_output_lora
        adjusted_up_output = up_output + up_output_lora

        down_output = down_proj(act_fn(adjusted_gate_output) * adjusted_up_output)
        down_output_lora = down_proj_lora_B(down_proj_lora_A(act_fn(adjusted_gate_output) * adjusted_up_output))
        # print('down', down_output.shape) #299, 1024]) [401, 1024])
        return down_output + down_output_lora

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        selected_experts_set = set(selected_experts.flatten().tolist())

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in selected_experts_set:
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            
            current_hidden_states = self.basenet_forward(current_state, expert_idx) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits
