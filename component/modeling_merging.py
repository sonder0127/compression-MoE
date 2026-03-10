import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock


# olmoe similarity   
class OlmoeSimilarityWrapper(nn.Module):
    def __init__(self, model, device):
        super(OlmoeSimilarityWrapper, self).__init__()
        self.model = model
        self.device = device
        self.expert_counts = {}
        self.expert_weights_sum = {}
        # 累加每个批次计算得到的 router_logits 相似性和专家输出相似性
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            # 仅对 MoE 层注册钩子（请替换为实际的 MoE 层类型）
            if isinstance(module, OlmoeSparseMoeBlock):
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            final_hidden_states, router_logits = output
            batch_size, seq_len, hidden_dim = final_hidden_states.shape
            total_samples = batch_size * seq_len
            num_experts = module.num_experts
            # reshape router_logits为 (total_samples, num_experts)
            router_logits = router_logits.view(-1, num_experts)

            # 对 router_logits 做 softmax，然后取 top_k 专家
            routing_weights_all = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights_all, module.top_k, dim=-1)
            if module.norm_topk_prob:
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            routing_weights = routing_weights.to(final_hidden_states.dtype)
            selected_experts = selected_experts.to(self.device)  # shape: (total_samples, top_k)
            # 初始化本层统计变量（仅第一次调用时）
            if layer_name not in self.expert_counts:
                self.expert_counts[layer_name] = torch.zeros(num_experts, device=self.device)
                self.expert_weights_sum[layer_name] = torch.zeros(num_experts, device=self.device)
                self.router_logits_similarity_sum[layer_name] = {
                    'cosine': torch.zeros((num_experts, num_experts), device=self.device),
                    'l2': torch.zeros((num_experts, num_experts), device=self.device)
                }
                self.router_logits_similarity_count[layer_name] = 0
                self.expert_output_similarity_sum[layer_name] = {
                    'cosine': torch.zeros((num_experts, num_experts), device=self.device),
                    'l2': torch.zeros((num_experts, num_experts), device=self.device)
                }
                self.expert_output_similarity_count[layer_name] = 0

            # 更新专家激活计数：每个样本中如果某个专家出现在 top_k 内，就记一次
            for i in range(total_samples):
                unique_experts = torch.unique(selected_experts[i])
                for expert in unique_experts:
                    self.expert_counts[layer_name][expert] += 1

            # 累加 routing weights：对每个样本，在 top_k 内，每个专家贡献其权重（可能多个位置出现时求和）
            for i in range(total_samples):
                for k in range(routing_weights.shape[1]):
                    expert = selected_experts[i, k]
                    self.expert_weights_sum[layer_name][expert] += routing_weights[i, k].item()
                    
            # 计算当前批次的 router_logits 相似性
            cosine_matrix_logits = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix_logits = torch.zeros((num_experts, num_experts), device=self.device)
            for i in range(num_experts):
                for j in range(i, num_experts):
                    cos_sim_logits = F.cosine_similarity(router_logits[:, i], router_logits[:, j], dim=0)
                    cosine_matrix_logits[i, j] = cos_sim_logits
                    cosine_matrix_logits[j, i] = cos_sim_logits
                    l2_dist_logits = torch.norm(router_logits[:, i] - router_logits[:, j], p=2)
                    l2_matrix_logits[i, j] = l2_dist_logits
                    l2_matrix_logits[j, i] = l2_dist_logits
            self.router_logits_similarity_sum[layer_name]['cosine'] += cosine_matrix_logits
            self.router_logits_similarity_sum[layer_name]['l2'] += l2_matrix_logits
            self.router_logits_similarity_count[layer_name] += 1
            
            hidden_states = input[0].view(-1, hidden_dim)
            expert_outputs = []
            for expert_idx in range(num_experts):
                expert_out = torch.zeros(total_samples, hidden_dim, device=self.device, dtype=final_hidden_states.dtype)
                for i in range(total_samples):
                    mask = (selected_experts[i] == expert_idx)
                    if mask.sum() > 0:
                        # 对于样本 i，该专家的贡献为该样本隐藏状态乘以该专家在 top_k 中所有位置的 routing weight之和
                        agg_weight = routing_weights[i][mask].sum()
                        # 模拟专家输出
                        expert_layer = module.experts[expert_idx]
                        current_state = hidden_states[i].unsqueeze(0)
                        current_output = expert_layer(current_state)
                        expert_out[i] = current_output.squeeze(0) * agg_weight
                expert_outputs.append(expert_out.view(-1))  # flatten为向量

            # 对专家输出进行中心化和归一化
            expert_outputs = torch.stack(expert_outputs)
            # 中心化
            expert_outputs = expert_outputs - expert_outputs.mean(dim=1, keepdim=True)
            # 归一化
            expert_outputs = F.normalize(expert_outputs, p=2, dim=1)

            # 计算当前批次的专家输出相似性
            cosine_matrix_output = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix_output = torch.zeros((num_experts, num_experts), device=self.device)
            for i in range(num_experts):
                for j in range(i, num_experts):
                    vec_i = expert_outputs[i]
                    vec_j = expert_outputs[j]
                    cos_sim_output = F.cosine_similarity(vec_i.unsqueeze(0), vec_j.unsqueeze(0), dim=1).item()
                    cosine_matrix_output[i, j] = cos_sim_output
                    cosine_matrix_output[j, i] = cos_sim_output
                    l2_dist_output = torch.norm(vec_i - vec_j, p=2)
                    l2_matrix_output[i, j] = l2_dist_output
                    l2_matrix_output[j, i] = l2_dist_output

            self.expert_output_similarity_sum[layer_name]['cosine'] += cosine_matrix_output
            self.expert_output_similarity_sum[layer_name]['l2'] += l2_matrix_output
            self.expert_output_similarity_count[layer_name] += 1

        return hook

    def reset_counts(self):
        self.expert_counts = {}
        self.expert_weights_sum = {}
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_similarity(self):
        router_logits_similarity = {}
        expert_output_similarity = {}
        for layer in self.router_logits_similarity_sum.keys():
            count_logits = self.router_logits_similarity_count[layer]
            cosine_matrix_logits = (self.router_logits_similarity_sum[layer]['cosine'] / count_logits).tolist()
            l2_matrix_logits = (self.router_logits_similarity_sum[layer]['l2'] / count_logits).tolist()
            router_logits_similarity[layer] = {'cosine': cosine_matrix_logits, 'l2': l2_matrix_logits}

            count_output = self.expert_output_similarity_count[layer]
            cosine_matrix_output = (self.expert_output_similarity_sum[layer]['cosine'] / count_output).tolist()
            l2_matrix_output = (self.expert_output_similarity_sum[layer]['l2'] / count_output).tolist()
            expert_output_similarity[layer] = {'cosine': cosine_matrix_output, 'l2': l2_matrix_output}
        return router_logits_similarity, expert_output_similarity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()



class OlmoeRouterLogitsSimilarityWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super( OlmoeRouterLogitsSimilarityWrapper, self).__init__()
        self.model = model
        self.device = device
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            _, router_logits = output
            num_experts = router_logits.shape[-1]
            router_logits = router_logits.view(-1, num_experts)

            cosine_matrix = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix = torch.zeros((num_experts, num_experts), device=self.device)

            for i in range(num_experts):
                for j in range(i, num_experts):
                    cos_sim = torch.nn.functional.cosine_similarity(router_logits[:, i], router_logits[:, j], dim=0)
                    cosine_matrix[i, j] = cos_sim
                    cosine_matrix[j, i] = cos_sim
                    l2_dist = torch.norm(router_logits[:, i] - router_logits[:, j], p=2)
                    l2_matrix[i, j] = l2_dist
                    l2_matrix[j, i] = l2_dist

            if layer_name not in self.router_logits_similarity_sum:
                self.router_logits_similarity_sum[layer_name] = {
                    'cosine': cosine_matrix,
                    'l2': l2_matrix
                }
                self.router_logits_similarity_count[layer_name] = 1
            else:
                self.router_logits_similarity_sum[layer_name]['cosine'] += cosine_matrix
                self.router_logits_similarity_sum[layer_name]['l2'] += l2_matrix
                self.router_logits_similarity_count[layer_name] += 1

        return hook

    def reset_counts(self):
        self.router_logits_similarity_sum = {}
        self.router_logits_similarity_count = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_router_logits_similarity(self):
        similarity = {}
        for layer in self.router_logits_similarity_sum.keys():
            count = self.router_logits_similarity_count[layer]
            cosine_matrix = self.router_logits_similarity_sum[layer]['cosine'] / count
            l2_matrix = self.router_logits_similarity_sum[layer]['l2'] / count
            similarity[layer] = {
                'cosine': cosine_matrix.tolist(),
                'l2': l2_matrix.tolist()
            }
        return similarity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class OlmoeOutputLogitsSimilarityWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super(OlmoeOutputLogitsSimilarityWrapper, self).__init__()
        self.model = model
        self.device = device
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):
                hook = module.register_forward_hook(self._expert_layer_forward_hook(name))
                self.hooks.append(hook)

    def _expert_layer_forward_hook(self, layer_name):
        def hook(module, input, output):
            final_hidden_states, _ = output
            num_experts = final_hidden_states.shape[-1]
            hidden_dim = final_hidden_states.shape[-2]
            final_hidden_states = final_hidden_states.view(-1, hidden_dim, num_experts)

            cosine_matrix = torch.zeros((num_experts, num_experts), device=self.device)
            l2_matrix = torch.zeros((num_experts, num_experts), device=self.device)

            for i in range(num_experts):
                for j in range(i, num_experts):
                    expert_i_output = final_hidden_states[..., i].view(-1)
                    expert_j_output = final_hidden_states[..., j].view(-1)
                    cos_sim = torch.nn.functional.cosine_similarity(expert_i_output, expert_j_output, dim=0)
                    cosine_matrix[i, j] = cos_sim
                    cosine_matrix[j, i] = cos_sim
                    l2_dist = torch.norm(expert_i_output - expert_j_output, p=2)
                    l2_matrix[i, j] = l2_dist
                    l2_matrix[j, i] = l2_dist

            if layer_name not in self.expert_output_similarity_sum:
                self.expert_output_similarity_sum[layer_name] = {
                    'cosine': cosine_matrix,
                    'l2': l2_matrix
                }
                self.expert_output_similarity_count[layer_name] = 1
            else:
                self.expert_output_similarity_sum[layer_name]['cosine'] += cosine_matrix
                self.expert_output_similarity_sum[layer_name]['l2'] += l2_matrix
                self.expert_output_similarity_count[layer_name] += 1

        return hook

    def reset_counts(self):
        self.expert_output_similarity_sum = {}
        self.expert_output_similarity_count = {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def compute_expert_similarity(self):
        similarity = {}
        for layer in self.expert_output_similarity_sum.keys():
            count = self.expert_output_similarity_count[layer]
            cosine_matrix = self.expert_output_similarity_sum[layer]['cosine'] / count
            l2_matrix = self.expert_output_similarity_sum[layer]['l2'] / count
            similarity[layer] = {
                'cosine': cosine_matrix.tolist(),
                'l2': l2_matrix.tolist()
            }
        return similarity

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
