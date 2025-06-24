import torch
import numpy as np
from tqdm import tqdm
from ..utils.data_utils import get_full_test_data, get_part_test_data
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