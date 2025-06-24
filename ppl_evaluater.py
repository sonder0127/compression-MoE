import torch
import numpy as np
from tqdm import tqdm
from utils.data_utils import get_full_test_data, get_part_test_data, get_test_data_local_part_datasets
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

# @torch.no_grad()
# def ppl_eval(model, tokenizer, num_samples, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
#     model.to(device)
#     model.eval()
#     ppls = {}
#     for dataset in datasets:
#         test_loader = get_test_data_local(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
#         nlls = []
#         for batch in tqdm(test_loader):
#             batch = batch.to(device)
#             output = model(batch, use_cache=False)
#             lm_logits = output.logits
#             if torch.isfinite(lm_logits).all():
#                 shift_logits = lm_logits[:, :-1, :].contiguous()
#                 shift_labels = batch[:, 1:].contiguous()
                
#                 loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
#                 loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
#                 nlls.append(loss)
#         ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
#         ppls[dataset] = ppl
#     print("PPL after pruning: {}".format(ppls))
#     print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

@torch.no_grad()
def ppl_eval(model, tokenizer, num_samples, datasets=['wikitext2', 'ptb'], model_seq_len=2048, batch_size=32, device="cuda", seed = 3):
    model.to(device)
    model.eval()
    ppls = {}

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # 获取测试数据加载器
        if num_samples != 0:
            test_loader = get_part_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size, num_samples=num_samples, seed=seed)
        else:
            test_loader = get_full_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        
        nlls = []
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset}"):
            batch = batch.to(device)
            
            # 确保输入批次有效
            if batch is None or batch.size(0) == 0:
                # print(f"Skipping empty batch in {dataset}")
                continue
            
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            
            # 确保模型输出有效
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            else:
                print(f"Invalid logits in {dataset}")

        if len(nlls) == 0:
            # print(f"No valid batches processed for {dataset}. Skipping...")
            continue
        
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
        print(f'{dataset} ppl', ppl)
        del nlls
        torch.cuda.empty_cache()
    print("PPL after pruning: {}".format(ppls))
    # 清理所有剩余的变量
    del loss_fct
    torch.cuda.empty_cache()
    return ppls

torch.no_grad()
def ppl_eval_low_source(model, tokenizer, num_samples, datasets=['wikitext2', 'ptb'], 
             model_seq_len=2048, batch_size=32, device="cuda", seed=3):
    model.to(device)
    model.eval()
    ppls = {}

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # 获取测试数据加载器
        if num_samples != 0:
            test_loader = get_part_test_data(dataset, tokenizer, seq_len=model_seq_len, 
                                             batch_size=batch_size, num_samples=num_samples, seed=seed)
        else:
            test_loader = get_full_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
        
        nlls = []
        for batch in tqdm(test_loader, desc=f"Evaluating {dataset}"):
            # 1. 确保batch有效并转移到device（避免无效数据占用显存）
            if batch is None or batch.size(0) == 0:
                continue
            batch = batch.to(device, non_blocking=True)  # 使用non_blocking加速数据传输
            
            # 2. 模型推理（禁用缓存减少显存）
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            del output  # 立即释放output对象（包含中间层输出）
            
            # 3. 计算loss并转移到CPU存储（减少GPU显存占用）
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            del lm_logits, batch  # 释放原始logits和batch
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            del shift_logits, shift_labels  # 释放中间变量
            
            nlls.append(loss.cpu())  # 将loss转移到CPU，避免GPU显存累积
            torch.cuda.empty_cache()  # 清理GPU缓存（可选，视显存压力决定）

        if not nlls:
            continue
        
        # 4. 合并loss并计算PPL（在CPU上处理）
        loss_tensor = torch.cat(nlls, dim=0)
        ppl = np.exp(loss_tensor.mean().item())
        ppls[dataset] = ppl
        print(f'{dataset} ppl: {ppl:.4f}')
        
        # 5. 释放当前dataset的所有临时数据
        del nlls, loss_tensor
        torch.cuda.empty_cache()

    print("PPL after pruning:", ppls)
    return ppls

@torch.no_grad()
def ppl_eval_part_datasets(model, tokenizer, num_samplis, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    ppls = {}

    for dataset in datasets:

        test_loader = get_test_data_local_part_datasets(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size, num_samples=num_samplis)
        
        nlls = []
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
        last_ppl = ppl  # 更新为当前数据集的困惑度
        
        # # 打印结果
        print(f"PPL after pruning on {dataset}: {ppl}")

    return last_ppl