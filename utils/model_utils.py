#coding:utf8
import os
import sys
import torch
import torch.nn as nn
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
)
from typing import Dict
from transformers import OlmoeForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

def get_model_from_huggingface(model_id):
    from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, LlamaForCausalLM
    if "opt" in model_id or "mistral" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16, trust_remote_code=True, cache_dir=None)
    model.seqlen = 2048
    return model, tokenizer

def get_model_saved_from_local(model_file, device):
    pruned_dict = torch.load(model_file, map_location=device)
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def get_MoE_model_from_local(device, model_name):
    if "OLMoE" in model_name:
        model_load_path = '../svd-moe/model/OLMoE'
        
        quantization_chioce = 0
        if quantization_chioce == 1:
            quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
            )
            
            model = OlmoeForCausalLM.from_pretrained(
                model_load_path,
                quantization_config=quantization_config,
                device_map=device
            )
        else:
            model = OlmoeForCausalLM.from_pretrained(
                model_load_path,
                local_files_only=True,  # 明确指定只使用本地文件
                device_map=device
            )
    elif model_name in ('Qwen', 'DeepSeek'):
        print('model_name')
        if 'Qwen' in model_name:
            # model_load_path = '../svd-moe/model/Qwen15-MoE-A27B'
            model_load_path = '../svd-moe/model/Qwen'
        elif 'DeepSeek' in model_name:
            # model_load_path = '../svd-moe/model/deepseek-moe-16b-base'
            model_load_path = '../svd-moe/model/DeepSeek'
            
        # 选择是否启用量化
        quantization_choice = 0  # 设置为1以启用4位量化
        
        if quantization_choice == 1:
            # 配置4位量化
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path,
                quantization_config=quantization_config,
                device_map=device,
                trust_remote_code=True  # 添加这一行以信任远程代码
            )
        else:
            # 不使用量化，直接从本地加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_load_path,
                local_files_only=True,  # 明确指定只使用本地文件
                device_map=device,
                trust_remote_code=True  # 添加这一行以信任远程代码
            )
        
        # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, local_files_only=True)
    
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def model_test(model, tokenizer, dev):
    # 定义一个简单的输入句子
    input_sentence = "Hello, who are you"

    # 使用 tokenizer 处理输入
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True).to(dev)

    # 获取模型的输出
    with torch.no_grad():  # 确保不计算梯度以节省内存和时间
        outputs = model(**inputs)

    # 打印 logits 或其他您感兴趣的输出部分
    #print("Model output logits:", outputs.logits)
    
    # 如果是生成任务，可以解码生成的token ids为文本
    if hasattr(outputs, 'logits'):
        predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
        decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        print("Decoded model output:", decoded_text)
    
def prepare_inputs_for_model(batch, tokenizer, device='cuda'):
    """
    准备模型输入数据。

    参数:
        batch: 来自数据加载器的一个批次数据。
        tokenizer: 用于处理文本输入的分词器（在此函数中可能不需要使用）。
        device: 设备类型（'cuda' 或 'cpu'），默认为 'cuda'。

    返回:
        dict: 包含模型所需输入的字典。
    """
    if not batch or not isinstance(batch, list) or not all(isinstance(item, dict) and 'input_ids' in item and 'attention_mask' in item for item in batch):
        raise ValueError("Batch should be a non-empty list of dictionaries containing 'input_ids' and 'attention_mask'.")
    
    # 合并批次中的所有 input_ids 和 attention_mask
    input_ids = torch.cat([item['input_ids'] for item in batch], dim=0).to(device)
    attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0).to(device)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def calculate_diff(layer1, layer2):
    diff_dict = {}
    # Ensure both layers have the same set of parameters
    param_names = set(layer1.state_dict().keys()) & set(layer2.state_dict().keys())
    for name in param_names:
        param1 = layer1.state_dict()[name]
        param2 = layer2.state_dict()[name]
        # Calculate the difference between the two parameters
        diff = param1 - param2
        # Store the difference in the dictionary
        diff_dict[name] = diff
    
    return diff_dict

def prepare_inputs_for_qwen(model, batch: Dict, 
                           tokenizer: AutoTokenizer,
                           device: str) -> Dict[str, torch.Tensor]:
    """Qwen专用输入预处理（修复梯度追踪警告版）"""
    padding_side = tokenizer.padding_side
    max_length = min(
        max(len(ids) for ids in batch["input_ids"]),
        tokenizer.model_max_length
    )
    
    # 修复点1：使用 clone().detach() 替代 torch.tensor()
    truncated_ids = [
        ids[:max_length].clone().detach()  # 断开梯度追踪
        for ids in batch["input_ids"]
    ]
    
    # 修复点2：优化attention_mask生成
    model_inputs = {
        "input_ids": pad_sequence(
            truncated_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).to(device, non_blocking=True),  # 异步传输
        
        "attention_mask": pad_sequence(
            [torch.ones_like(ids) for ids in truncated_ids],  # 直接复用已截断的ids形状
            batch_first=True,
            padding_value=0
        ).to(device, non_blocking=True)
    }
    
    # 位置编码处理（保持原逻辑）
    if hasattr(model.config, "use_position_embeddings") and model.config.use_position_embeddings:
        seq_length = model_inputs["input_ids"].size(1)
        position_ids = torch.arange(
            seq_length, 
            dtype=torch.long, 
            device=device  # 确保与输入设备一致
        ).unsqueeze(0)
        model_inputs["position_ids"] = position_ids.expand_as(model_inputs["input_ids"])
    
    return model_inputs

def save_expert_importance(expert_importance, output_dir='outfiles', filename='expert_importance.json'):
    """
    将专家重要性保存到 JSON 文件中。

    参数:
        expert_importance: 包含每一层专家重要性的字典。
        output_dir: 输出目录，默认为 'outfiles'。
        filename: 输出文件名，默认为 'expert_importance.json'。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整路径
    output_path = os.path.join(output_dir, filename)

    # 保存为 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(expert_importance, f, indent=4)

    print(f"Expert importance saved to {output_path}")

def save_all_expert_similarity(router_logits_similarity, output_similarity, output_dir='outfile/expert_importance'):
    """
    将 router logits、expert similarity 和 output similarity 保存到 JSON 文件中。
    
    参数:
        router_logits: 每一层的 router logits。
        expert_similarity: 基于 router logits 的专家相似性矩阵。
        output_similarity: 基于专家输出的专家相似性矩阵。
        output_dir: 输出目录路径，默认为 'outfiles/expert_importance'。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 构建完整路径
    expert_similarity_path = os.path.join(output_dir, 'router_logits_similarity.json')
    output_similarity_path = os.path.join(output_dir, 'output_similarity.json')

    # 将 numpy 数组或 PyTorch 张量转换为列表
    def convert_to_list(data):
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return data.tolist()
        elif isinstance(data, list):
            return [convert_to_list(item) for item in data]
        else:
            return data

    # 将 expert_similarity 转换为列表
    router_logits_similarity_list = {layer: {'cosine': convert_to_list(sim_matrix['cosine']), 
                                      'l2': convert_to_list(sim_matrix['l2'])} 
                              for layer, sim_matrix in router_logits_similarity.items()}

    # 将 output_similarity 转换为列表
    output_similarity_list = {layer: {'cosine': convert_to_list(sim_matrix['cosine']), 
                                      'l2': convert_to_list(sim_matrix['l2'])} 
                              for layer, sim_matrix in output_similarity.items()}

    with open(expert_similarity_path, 'w') as f:
        json.dump(router_logits_similarity_list, f, indent=4)
    with open(output_similarity_path, 'w') as f:
        json.dump(output_similarity_list, f, indent=4)

def save_expert_similarity(similarity, file_name, output_dir='outfile/expert_importance'):
    """
    将 router logits、expert similarity 和 output similarity 保存到 JSON 文件中。
    
    参数:
        router_logits: 每一层的 router logits。
        expert_similarity: 基于 router logits 的专家相似性矩阵。
        output_similarity: 基于专家输出的专家相似性矩阵。
        output_dir: 输出目录路径，默认为 'outfiles/expert_importance'。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    output_similarity_path = os.path.join(output_dir, file_name)

    # 将 numpy 数组或 PyTorch 张量转换为列表
    def convert_to_list(data):
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return data.tolist()
        elif isinstance(data, list):
            return [convert_to_list(item) for item in data]
        else:
            return data

    # 将 expert_similarity 转换为列表
    similarity_list = {layer: {'cosine': convert_to_list(sim_matrix['cosine']), 
                                      'l2': convert_to_list(sim_matrix['l2'])} 
                              for layer, sim_matrix in similarity.items()}

    with open(output_similarity_path, 'w') as f:
        json.dump(similarity_list, f, indent=4)


