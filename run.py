import os
import json
import gc
import torch
import time

from utils.model_utils import get_MoE_model_from_local, parse_order, dict_to_namespace, append_result_to_file, safe_int
from ppl_evaluater import ppl_inference_eval
from acc_evaluator import run_lm_eval
from calcute import solve_pmd_any_order
from compress_method import apply_compression_by_order

#======================== 剪枝的方法与超参数 ===========================
# args.pruning.strategy: threshold, fixed
# args.pruning.importance_metrics: activation_frequency, sum_routing_weights, input_l2_avg, ouput_l2_avg

#======================== 合并的方法与超参数 ===========================
# args.merging.eval_object: output, router_logits, weight
# args.merging.metrics: l2, cosine
# args.merging.weighting_factor: activation_frequency, sum_routing_weights, input_l2_avg, ouput_l2_avg

#======================== 低秩分解的方法与超参数 ===========================
# args.svd.method: vanilla-SVD, SVD-LLM, ASVD

if __name__ == "__main__":
    config_file = "config/PMD.json"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    args = dict_to_namespace(config)

    device = args.DEV
    order = parse_order(getattr(args, "order", None))
    print("Compression order =", order)

    #本地模型文件地址
    model_load_path = {
        "OLMoE": './model/OLMoE',
        "Qwen": './model/Qwen'
    }
    result_save_dir = f"outfile/{args.model_name}/{args.dataset}/evaluation_results"
    os.makedirs(result_save_dir, exist_ok=True)
    order_str = "".join(order)
    result_filename = f"order={order_str}_pruning={args.pruning.strategy}+{args.pruning.importance_metrics}_merging={args.merging.eval_object}+{args.merging.metrics}+{args.merging.weighting_factor}_svd={args.svd.method}s_eed={args.seed}.json"
    result_file_path = os.path.join(result_save_dir, result_filename)

    start_time = time.time()

    
    #压缩率
    all_ratio = 0.5
    #三类方法的贡献占比
    pruning_ratio = 0
    merging_ratio = 0
    svd_ratio = 1-pruning_ratio-merging_ratio

    param_ratio = {}
    param_ratio['pruning_ratio'] = pruning_ratio
    param_ratio['merging_ratio'] = merging_ratio
    param_ratio['svd_ratio'] = svd_ratio

    # ===== 根据压缩率和贡献率计算每种方法的压缩参数 =====
    compress_param = solve_pmd_any_order(
        all_ratio=all_ratio,
        pruning_ratio=pruning_ratio,
        merging_ratio=merging_ratio,
    )
    # 总的剪枝专家个数
    num_experts_pruning = safe_int(compress_param.get("num_pruning", 0), default=0)
    # 合并后专家应保留的个数
    num_expert_group   = safe_int(compress_param.get("num_group", 0), default=0)
    # 低秩分解的lora rank
    lora_rank          = safe_int(compress_param.get("svd_rank", 0), default=0)

    model, tokenizer = get_MoE_model_from_local(device, args.model_name, model_load_path[args.model_name])
    # # ===== 根据 order 执行压缩流水线 =====
    # model = apply_compression_by_order(
    #     args=args,
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     order=order,
    #     num_experts_pruning=num_experts_pruning,
    #     num_expert_group=num_expert_group,
    #     lora_rank=lora_rank,
    #     param_ratio=param_ratio
    # )
    
    if args.save_model:
        order_str = "".join(order)
        filename = f'order={order_str}_all_ratio={all_ratio}_pruning_ratio={pruning_ratio}_merging_ratio={merging_ratio}_svd_ratio={svd_ratio}_seed={args.seed}.pt'
        save_dir = f'./outfile/{args.model_name}/{args.dataset}/model_saved'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, filename)
        torch.save({'model': model, 'tokenizer': tokenizer}, model_save_path)

    # ===== 评测 =====
    if getattr(args.eval, "enabled", False):
        # device = 'cuda'
        model.eval()
        model = model.to(device)

        ppl, inference_mertrics = ppl_inference_eval(
            model,
            tokenizer,
            args.eval.nsamples,
            datasets=["wikitext2", "ptb"],
            model_seq_len=args.eval.model_seq_len,
            batch_size=args.eval.batch_size,
            device=device,
        )
        print(ppl)
        if isinstance(ppl, dict):
            ppl_dict = ppl
        elif isinstance(ppl, (float, int)):
            ppl_dict = {"wikitext2": float(ppl), "ptb": float(ppl)}
        else:
            ppl_dict = {"wikitext2": "N/A", "ptb": "N/A"}
        
        acc = run_lm_eval(
            model,
            args.eval.nsamples,
            batch_size=16,
            task_names=["openbookqa", "mathqa", "arc_easy", "arc_challenge"],
        )
        print(acc)
        order_str = "".join(order)
        key = f"order={order_str}_all_ratio={all_ratio}_pruning_ratio={pruning_ratio}_merging_ratio={merging_ratio}_svd_ratio={svd_ratio}_seed={args.seed}"
        result = {
            key: {
                "all_ratio": all_ratio,
                "pruning_ratio": pruning_ratio,
                "merging_ratio": merging_ratio,
                "svd_ratio": svd_ratio,
                "order": "".join(order),
                "compress_param": compress_param,
                "perplexity": {
                    "wikitext2": ppl_dict.get("wikitext2", "N/A"),
                    "ptb": ppl_dict.get("ptb", "N/A"),
                },
                "accuracy": {
                    "arc_easy": acc.get("arc_easy", {}).get("acc", "N/A"),
                    "arc_challenge": acc.get("arc_challenge", {}).get("acc", "N/A"),
                    "mathqa": acc.get("mathqa", {}).get("acc", "N/A"),
                    "openbookqa": acc.get("openbookqa", {}).get("acc", "N/A"),
                },
                "inference_mertrics": inference_mertrics,
            }
        }
        append_result_to_file(result, result_file_path)
    

    # ===== 清理 =====
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds.")
    print(f"Evaluation results saved to {result_file_path}")
