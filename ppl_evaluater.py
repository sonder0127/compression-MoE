import torch
import numpy as np
import time
from tqdm import tqdm
from utils.data_utils import get_full_test_data, get_part_test_data
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)



@torch.no_grad()
def ppl_inference_eval(
    model,
    tokenizer,
    num_samples,
    datasets=("wikitext2", "ptb"),
    model_seq_len=2048,
    batch_size=32,
    device="cuda",
):
    """
    输出：
    metrics = {
        "<dataset>": {
            "ppl": float,
            "num_batches": int,
            "total_samples": int,
            "total_tokens": int,
            "total_time_s": float,
            "avg_latency_s_per_batch": float,
            "p50_latency_s": float,
            "p95_latency_s": float,
            "throughput_tokens_per_s": float,
            "throughput_samples_per_s": float,
            "memory": {
                "start_alloc_bytes": int,
                "start_reserved_bytes": int,
                "peak_alloc_bytes": int,
                "peak_reserved_bytes": int,
                "end_alloc_bytes": int,
                "end_reserved_bytes": int,
            }  # 若非cuda，则为None
        }
    }
    return {"metrics": metrics}
    """

    def _is_cuda(dev: str) -> bool:
        return isinstance(dev, str) and dev.startswith("cuda") and torch.cuda.is_available()

    def _mem_snapshot():
        # bytes
        return {
            "alloc": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
        }

    model.to(device)
    model.eval()

    metrics = {}

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        # 获取测试数据加载器
        if num_samples != 0:
            test_loader = get_part_test_data(
                dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size, num_samples=num_samples
            )
        else:
            test_loader = get_full_test_data(
                dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size
            )

        use_cuda_mem = _is_cuda(device)

        # 显存统计初始化
        if use_cuda_mem:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_mem = _mem_snapshot()
        else:
            start_mem = None

        # NLL 统计（避免保存所有 token loss 导致内存爆炸）
        nll_sum = 0.0
        token_count = 0

        # 吞吐/延迟统计
        batch_latencies = []
        total_samples = 0
        total_tokens = 0
        num_batches = 0

        t_dataset_start = time.perf_counter()

        for batch in tqdm(test_loader, desc=f"Evaluating {dataset}"):
            # 你的 dataloader 产出通常是 tensor: [B, T]
            if batch is None:
                continue

            batch = batch.to(device)

            if not torch.is_tensor(batch) or batch.dim() < 2 or batch.size(0) == 0:
                continue

            # 计数：samples/tokens
            B, T = int(batch.size(0)), int(batch.size(1))
            # ppl 用 shift (T-1) 个 token
            cur_tokens = B * max(T - 1, 0)

            # 计时（建议同步避免 GPU 异步导致 latency 偏小）
            if use_cuda_mem:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            output = model(batch, use_cache=False)
            lm_logits = output.logits

            # loss（sum）
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                # sum over all tokens in this batch
                loss_fct_sum = torch.nn.CrossEntropyLoss(reduction="sum")
                loss_sum = loss_fct_sum(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                nll_sum += float(loss_sum.item())
                token_count += int(shift_labels.numel())
            else:
                print(f"Invalid logits in {dataset}, skip this batch.")
                # 即使 skip，也把时间记录下来更公平；但 tokens/s 不计入
                cur_tokens = 0
                B = 0

            if use_cuda_mem:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latency = t1 - t0
            batch_latencies.append(latency)

            if cur_tokens > 0 and B > 0:
                total_samples += B
                total_tokens += cur_tokens
                num_batches += 1

        t_dataset_end = time.perf_counter()
        total_time_s = t_dataset_end - t_dataset_start

        if token_count == 0 or num_batches == 0:
            print(f"No valid batches processed for {dataset}. Skipping...")
            # 结束显存快照
            if use_cuda_mem:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            continue

        # ppl
        mean_nll = nll_sum / max(token_count, 1)
        ppl = float(np.exp(mean_nll))

        # latency stats
        avg_latency = float(total_time_s / max(num_batches, 1))
        p50_latency = float(np.percentile(batch_latencies, 50)) if batch_latencies else 0.0
        p95_latency = float(np.percentile(batch_latencies, 95)) if batch_latencies else 0.0

        # throughput
        throughput_tokens = float(total_tokens / max(total_time_s, 1e-12))
        throughput_samples = float(total_samples / max(total_time_s, 1e-12))

        # memory stats
        if use_cuda_mem:
            peak_alloc = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
            end_mem = _mem_snapshot()
            memory_block = {
                "start_alloc_bytes": int(start_mem["alloc"]),
                "start_reserved_bytes": int(start_mem["reserved"]),
                "peak_alloc_bytes": int(peak_alloc),
                "peak_reserved_bytes": int(peak_reserved),
                "end_alloc_bytes": int(end_mem["alloc"]),
                "end_reserved_bytes": int(end_mem["reserved"]),
            }
        else:
            memory_block = None

        metrics[dataset] = {
            "num_batches": int(num_batches),
            "total_samples": int(total_samples),
            "total_tokens": int(total_tokens),
            "total_time_s": float(total_time_s),
            "avg_latency_s_per_batch": float(avg_latency),
            "p50_latency_s": float(p50_latency),
            "p95_latency_s": float(p95_latency),
            "throughput_tokens_per_s": float(throughput_tokens),
            "throughput_samples_per_s": float(throughput_samples),
            "memory": memory_block,
        }

        # 清理
        torch.cuda.empty_cache()

    print("Metrics:", {"metrics": metrics})
    return ppl, metrics


@torch.no_grad()
def ppl_eval(model, tokenizer, num_samples, datasets=['wikitext2', 'ptb'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.to(device)
    model.eval()
    ppls = {}

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # 获取测试数据加载器
        if num_samples != 0:
            test_loader = get_part_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size, num_samples=num_samples)
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
        del nlls
        torch.cuda.empty_cache()
    print("PPL after pruning: {}".format(ppls))
    # 清理所有剩余的变量
    del loss_fct
    torch.cuda.empty_cache()
    return ppls