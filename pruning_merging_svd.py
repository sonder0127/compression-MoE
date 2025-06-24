import os
import json
from types import SimpleNamespace
import torch.jit
import torch
from utils.data_utils import get_calib_train_data_old
from utils.model_utils import get_MoE_model_from_local, model_test, save_expert_importance, save_expert_similarity
from expert_pruning import model_expert_pruning
from method.ppl_evaluater import ppl_eval
from compress_method import evaluate_model_expert_importance, evaluate_expert_param_similarity, \
    evaluate_expert_similarity, model_expert_merging_leading_for_all, profile_svdllm, profile_asvd_moe, profile_svdllm_low_source, profile_asvd_low_source, model_expert_only_SVD
from method.calcute import *
from method.acc_evaluator import run_lm_eval
import gc
import time

def dict_to_namespace(d):
    """递归地将字典转换为 SimpleNamespace 对象"""
    if isinstance(d, dict):
        namespace = SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return namespace
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d
def append_result_to_file(result, file_path):
    """将新结果追加到 JSON 文件中"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}
    existing_results.update(result)
    with open(file_path, 'w') as f:
        json.dump(existing_results, f, indent=4)


if __name__ == '__main__':
    # 直接从配置文件加载参数
    config_file = "config/pruning_merging_svd.json"  # 可根据需要修改配置文件路径
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    with open(config_file, "r") as f:
        config = json.load(f)
    args = dict_to_namespace(config)
    
    result_save_dir = f'outfile/{args.model_name}/{args.dataset}/evaluation_results'
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)
    result_filename = f'pruning_merging_svd_seed={args.seed}.json'
    result_file_path = os.path.join(result_save_dir, result_filename)
    device = 'cpu'
    start_time = time.time()
    for all_ratio in range(1, 10, 2):
        all_ratio /= 10.0
        print('all_ratio', all_ratio)
        for pr in range(1, 10, 2):
            res = 10-pr
            for mr in range(1, res+1, 2):
                sr = res-mr
                if 0 in [pr, mr, sr]:
                    continue
                print('pruning_ratio', pr/10.0, 'merging_ratio', mr/10.0, 'svd', sr/10.0)
                pruning_ratio = pr/ 10.0
                merging_ratio = mr/ 10.0
                svd_ratio = sr/ 10.0

                num_experts_pruning, num_expert_group, lora_rank = pruning_merging_svd_cal(all_ratio, pruning_ratio, merging_ratio)
                num_experts_pruning = int(num_experts_pruning)
                lora_rank = int(lora_rank)
                num_expert_group = int(num_expert_group)
                print(all_ratio, pruning_ratio, num_experts_pruning, num_expert_group, lora_rank)
                model, tokenizer = get_MoE_model_from_local(args.DEV, args.model_name)

                if args.pruning.enabled is True:
                    #======================== param ===========================
                    # args.pruning.strategy: threshold, fixed
                    # args.pruning.importance_metrics: activation_frequency, sum_routing_weights, input_l2_avg, ouput_l2_avg
                    if args.pruning.load_from_file is False:
                        cali_white_data = get_calib_train_data_old(args.dataset, tokenizer, args.pruning.eval_nsamples, args.eval.model_seq_len, args.seed)
                        expert_importance_data = evaluate_model_expert_importance(args.model_name, model, tokenizer, cali_white_data, device)
                        activation_frequency = expert_importance_data['activation_frequency']
                        sum_routing_weights = expert_importance_data['sum_routing_weights']
                        input_l2_avg = expert_importance_data['input_l2_avg']
                        output_l2_avg = expert_importance_data['output_l2_avg']

                        save_dir = f'../outfile/{args.model_name}/{args.dataset}/expert_importance'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        save_expert_importance(
                            expert_importance_data['activation_frequency'], save_dir, f'activation_frequency_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json')
                        save_expert_importance(
                            expert_importance_data['sum_routing_weights'], save_dir, f'sum_routing_weights_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json')
                        save_expert_importance(
                            expert_importance_data['input_l2_avg'], save_dir, f'input_l2_avg_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json')
                        save_expert_importance(
                            expert_importance_data['output_l2_avg'], save_dir, f'output_l2_avg_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json')
                        save_expert_importance(
                            expert_importance_data['contribution_sum'], save_dir, f'contribution_sum_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json')
                        experts_importance = expert_importance_data[f'{args.pruning.importance_metrics}']   
                        exit(0)
                    else:
                        save_dir = f'../outfile/{args.model_name}/{args.dataset}/expert_importance'
                        with open(f'{save_dir}/{args.pruning.importance_metrics}_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json', 'r') as file:
                            experts_importance = json.load(file)
                        print('load')
                
                    model, all_original_expert_indices = model_expert_pruning(
                        device, args.model_name, model, 
                        experts_importance, 
                        args.pruning.strategy, 
                        num_experts_pruning
                        )
                    if args.pruning.save_model:
                        filename = f'Dataset={args.dataset}_Pruned={num_experts_pruning}_Seed={args.seed}.pt'
                        save_dir = f'outfile/{args.model_name}/{args.dataset}/model_saved'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        model_save_path = os.path.join(save_dir, filename)
                        torch.save({'model': model, 'tokenizer': tokenizer}, model_save_path)
                if args.pruning.enabled is True:
                    if args.pruning.load_from_file is False:
                        del cali_white_data, expert_importance_data, activation_frequency, sum_routing_weights, input_l2_avg, output_l2_avg
                    del experts_importance
                            
                if args.merging.enabled is True:
                    #======================== param ===========================
                    # args.merging.eval_object: output, router_logits, weight
                    # args.merging.metrics: l2, cosine
                    # args.merging.weighting_factor: activation_frequency, sum_routing_weights, input_l2_avg, ouput_l2_avg
                    if args.merging.load_from_file is False:
                        save_dir = f'../outfile/{args.model_name}/{args.dataset}/expert_similarity'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        if args.merging.eval_object == 'weight':
                            expert_similarity = evaluate_expert_param_similarity(args.model_name, model, args.DEV)
                            print('similarity ok')
                            #save_expert_similarity(expert_similarity, f'weight_similarity.json', save_dir)
                        else:
                            cali_white_data = get_calib_train_data_old(args.dataset, tokenizer, args.merging.eval_nsamples, args.eval.model_seq_len, args.seed)
                            router_logits_similarity, output_similarity = evaluate_expert_similarity(model, tokenizer, cali_white_data, args.DEV)
                            save_expert_similarity(output_similarity, f'output_similarity_nsamples={args.merging.eval_nsamples}_seed={args.seed}.json', save_dir)
                            save_expert_similarity(router_logits_similarity, f'router_logits_similarity_nsamples={args.merging.eval_nsamples}_seed={args.seed}.json', save_dir)
                            if args.merging.eval_object == 'output':
                                expert_similarity = output_similarity
                            else:
                                expert_similarity = router_logits_similarity
                    else: 
                        save_dir = f'../outfile/{args.model_name}/{args.dataset}/expert_similarity'
                        if args.merging.eval_object != 'weight':
                            with open(f'{save_dir}/{args.merging.eval_object}_similarity_nsamples={args.merging.eval_nsamples}_seed={args.seed}.json', 'r') as file:
                                expert_similarity = json.load(file)
                        else:
                            with open(f'{save_dir}/{args.merging.eval_object}_similarity.json', 'r') as file:
                                expert_similarity = json.load(file)
                    
                    cali_white_data = get_calib_train_data_old(args.dataset, tokenizer, args.merging.eval_nsamples, args.eval.model_seq_len, args.seed)
                    expert_importance_data = evaluate_model_expert_importance(args.model_name, model, tokenizer, cali_white_data, device)
                    expert_importance = expert_importance_data['activation_frequency']
                    
                    # save_dir = f'../outfile/{args.model_name}/{args.dataset}/expert_importance'
                    # with open(f'{save_dir}/{args.merging.weighting_factor}_nsamples={args.pruning.eval_nsamples}_seed={args.seed}.json', 'r') as file:
                    #     expert_importance = json.load(file)

                    model_expert_merging_leading_for_all(
                        args.model_name, 
                        model, 
                        expert_importance, 
                        expert_similarity, 
                        args.merging.metrics,
                        num_expert_group, 
                        args.DEV,
                        all_original_expert_indices
                    )

                if args.merging.enabled is True:
                    if args.merging.load_from_file is False:
                        if args.merging.eval_object != 'weight':
                            del cali_white_data, router_logits_similarity, output_similarity
                    del expert_similarity, expert_importance
                    
                if args.svd.enabled is True:
                    #======================== param ===========================
                    # args.svd.method: vanilla-SVD, SVD-LLM, ASVD
                    if args.svd.method != 'vanilla-SVD':
                        if args.svd.load_from_file is False:
                            cali_white_data = get_calib_train_data_old(args.dataset, tokenizer, args.svd.whitening_nsamples, seqlen=args.eval.model_seq_len)
                            if args.svd.method == 'SVD-LLM':
                                profiling_mat = profile_svdllm(model, cali_white_data, device)
                            elif args.svd.method == 'ASVD':
                                profiling_mat = profile_asvd_moe(model, cali_white_data, device, alpha=0.5)
                            del cali_white_data
                            # save_dir = f'outfile/{args.model_name}/{args.dataset}/profiling'
                            # if not os.path.exists(save_dir):
                            #     os.makedirs(save_dir, exist_ok=True)
                            # save_path = os.path.join(save_dir, f'pruning={num_experts_pruning}_svd_method={args.svd.method}_nsamples={args.svd.whitening_nsamples}_seed={args.seed}.pt')
                            # torch.save(profiling_mat, save_path)
                        else:
                            #profiling_mat = torch.load(f'outfile/{args.model_name}/{args.dataset}/profiling/pruning={num_experts_pruning}_svd_method={args.svd.method}_nsamples={args.svd.whitening_nsamples}_seed={args.seed}.pt')
                            print('load success')
                    else:
                        profiling_mat = None
                    
                    model_expert_only_SVD(args.model_name, model, args.svd.method, args.svd.compress_ratio, lora_rank, args.DEV, profiling_mat)
                    if args.svd.save_model:
                        filename = f'merging={num_expert_group}_svd={args.svd.method}_rank={lora_rank}_seed={args.seed}.pt'
                        save_dir = f'../outfile/{args.model_name}/{args.dataset}/model_saved/merging/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        model_save_path = os.path.join(save_dir, filename)
                        print(model_save_path)
                        torch.save({'model': model, 'tokenizer': tokenizer}, model_save_path)

                    del profiling_mat

                if args.eval.enabled is True:
                    model.eval()
                    model = model.float()
                    model = model.to(args.DEV)
                    ppl = ppl_eval(model, tokenizer, args.eval.wiki_nsamples, datasets=["wikitext2", "ptb"],
                                                        model_seq_len=args.eval.model_seq_len,
                                                        batch_size=args.eval.wiki_batch_size, device=args.DEV)
                    print('ppl', ppl)
                    acc = run_lm_eval (model, args.eval.nsamples, batch_size=args.eval.batch_size, task_names=["openbookqa", "mathqa", "arc_easy", "arc_challenge"] )
                    # 保存结果
                    print('ppl', acc)
                    key = f"all_ratio={all_ratio}_pruning_ratio={pruning_ratio}_merging_ratio={merging_ratio}"
                    result = {
                        key: {
                            "all_ratio": all_ratio,
                            "pruning_ratio": pruning_ratio,
                            "merging_ratio": merging_ratio,
                            "svd_ratio": svd_ratio,
                            "perplexity": {
                                "wikitext": ppl.get('wikitext2', 'N/A'),
                                "ptb": ppl.get('ptb', 'N/A')
                            },
                            "accuracy": {
                                "arc_easy": acc.get('arc_easy', {}).get('acc', 'N/A'),
                                "arc_challenge":  acc.get('arc_challenge', {}).get('acc', 'N/A'),
                                "mathqa":  acc.get('mathqa', {}).get('acc', 'N/A'),
                                "openbookqa": acc.get('openbookqa', {}).get('acc', 'N/A')
                            }
                        }
                    }
                    append_result_to_file(result, result_file_path)
                # 释放显存和内存
                del model
                del tokenizer
                torch.cuda.empty_cache()
                gc.collect()
                
        end_time = time.time()
        # 计算并记录执行时间
        elapsed_time = end_time - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds.")

        print(f"Evaluation results saved to {result_file_path}")
