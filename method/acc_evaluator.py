import os
import torch
import numpy as np
import sys
sys.path.append('./lm-evaluation-harness')
os.environ["HF_ALLOW_CODE_EVAL"] = "1"  # 必须在所有huggingface/evaluate导入前
from lm_eval import tasks, evaluator
from lm_eval.models.huggingface import HFLM

def simplify_results(results):
    simplified = {}
    # 定义不同数据集关心的指标（新增 hellaswag 的 acc 和 acc_norm）
    dataset_metrics = {
        "wikitext": ["perplexity"],
        "winogrande": ["acc", "acc_stderr"],  
        "mbpp": ["pass_at_1", "acc_stderr"],
        "hellaswag": ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"], 
        "mathqa": ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"], 
        "arc_easy": ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"], 
        "arc_challenge": ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"] ,
        "openbookqa": ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"]
    }

    for task_name, task_result in results["results"].items():
        metrics = {}
        # 获取该数据集关心的指标
        if task_name in dataset_metrics:
            target_metrics = dataset_metrics[task_name]
            for metric_name, metric_value in task_result.items():
                # 排除标准误差指标（如 acc_stderr）
                if "stderr" in metric_name:
                    continue
                # 检查是否为目标指标
                for target_metric in target_metrics:
                    if target_metric in metric_name:
                        # 清洗指标名称（去除逗号后的后缀）
                        clean_metric_name = metric_name.split(',')[0]
                        metrics[clean_metric_name] = metric_value
        if metrics:
            simplified[task_name] = metrics
    return simplified
    
def run_lm_eval(model, limit, batch_size=256, task_names=["openbookqa", "arc_easy", "winogrande", "hellaswag",
             "arc_challenge", "piqa", "mathqa"]):

    wrapped_model = HFLM(pretrained=model)
    if limit != 0:
        results = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=task_names,
            batch_size=batch_size,
            device=next(model.parameters()).device,
            write_out=True,
            log_samples=True,
            num_fewshot=0,
            task_manager=tasks.TaskManager(),
            limit=limit  # 添加 limit 参数
        )
    else:
        results = evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=task_names,
            batch_size=batch_size,
            device=next(model.parameters()).device,
            write_out=True,
            log_samples=True,
            num_fewshot=0,
            task_manager=tasks.TaskManager()
        )
    

    # Remove samples from results to reduce file size
    if 'samples' in results:
        del results['samples']

    # Custom JSON Encoder to handle torch.Tensor, torch.device, and numpy.ndarray
    class CustomEncoder:
        def transform(self, obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            elif isinstance(obj, torch.device):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.dtype):
                return str(obj)
            elif isinstance(obj, dict):
                return {key: self.transform(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self.transform(item) for item in obj]
            return obj

    encoder = CustomEncoder()
    transformed_results = encoder.transform(results)
    simple_output = simplify_results(transformed_results)
    # 清理不再需要的变量以释放显存
    del wrapped_model, results, encoder, transformed_results
    torch.cuda.empty_cache()
    return simple_output