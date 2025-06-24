from torch import nn
from transformers.activations import ACT2FN
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.activations import ACT2FN

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

# class DeepseekMLP(nn.Module):
#     def __init__(self, config, ratio=1, lora_rank=602, hidden_size = None, intermediate_size = None):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
#         self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
#         self.lora_rank = lora_rank
        
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(self, x):
#         if self.config.pretraining_tp > 1:
#             slice = self.intermediate_size // self.config.pretraining_tp
#             gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
#             up_proj_slices = self.up_proj.weight.split(slice, dim=0)
#             down_proj_slices = self.down_proj.weight.split(slice, dim=1)

#             gate_proj = torch.cat(
#                 [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
#             )
#             up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

#             intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
#             down_proj = [
#                 F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
#             ]
#             down_proj = sum(down_proj)
#         else:
#             down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

#         return down_proj

# basenet==========================================================================================

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