import os
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import random
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# 禁用tokenizers并行处理（必须在导入tokenizers之前设置）
os.environ["TOKENIZERS_PARALLELISM"] = "false"
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

#============================================#
#              获取代理数据集                #
class CalibrationDataset(Dataset):
    def __init__(self, samples, tokenizer, seqlen):
        self.samples = samples
        self.tokenizer = tokenizer
        self.seqlen = seqlen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.seqlen,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0]
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch])
    }


def _normalize_dataset_name(name: str) -> str:
    """
    把各种写法归一到 {ptb, wikitext2}
    """
    n = (name or "").strip().lower()
    alias_map = {
        "ptb": "ptb",
        "penn": "ptb",
        "penn_treebank": "ptb",
        "treebank": "ptb",
        "penn-treebank": "ptb",

        "wikitext": "wikitext2",
        "wikitext2": "wikitext2",
        "wiki": "wikitext2",
        "wikitext-2": "wikitext2",
        "wikitext-2-raw": "wikitext2",
        "wikitext-2-raw-v1": "wikitext2",
    }
    # 允许你传 "wikitext-2-raw-v1" 等
    if n in alias_map:
        return alias_map[n]
    # 兜底：包含关键词就映射
    if "wiki" in n:
        return "wikitext2"
    if "ptb" in n or "penn" in n or "treebank" in n:
        return "ptb"
    raise ValueError(f"Unknown dataset name: {name}. Use 'ptb' or 'wikitext2' (or their aliases).")


def _dataset_config_by_name(name: str) -> dict:
    """
    你本地数据路径的配置集中在这
    """
    ds = _normalize_dataset_name(name)
    if ds == "ptb":
        return {
            "type": "text",
            "path": "./utils/datasets/ptb/ptb.train.txt",
            "text_field": "text",
        }
    # wikitext2
    return {
        "type": "parquet",
        "path": "./utils/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "text_field": "text",
    }


def _make_loader_from_tensors(input_ids: torch.Tensor,
                              attention_mask: torch.Tensor,
                              batch_size: int) -> DataLoader:
    """
    把缓存的 tensor 变回 DataLoader（batch 为 dict），与合并代码完全匹配
    """
    ds = TensorDataset(input_ids, attention_mask)

    def _collate(tuples):
        ii = torch.stack([t[0] for t in tuples])
        am = torch.stack([t[1] for t in tuples])
        return {"input_ids": ii, "attention_mask": am}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, pin_memory=True)

def get_calib_train_data(
    name: str,
    tokenizer,
    nsamples: int,
    seqlen: int = 2048,
    seed: int = 3,
    batch_size: int = 1,
    cache_dir: str = "cache",
) -> DataLoader:
    """
    nsamples:
      - >0: 随机不放回抽取 nsamples 条
      - 0 : 使用全量数据集
    """
    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(cache_dir, exist_ok=True)

    ds_key = _normalize_dataset_name(name)

    # ✅ nsamples=0 表示全量，缓存文件名也要区分
    ns_key = "all" if (nsamples == 0) else str(nsamples)
    cache_file = os.path.join(cache_dir, f"{ds_key}_{ns_key}_{seqlen}_{seed}_{batch_size}.pt")

    # 1) 读缓存
    if os.path.exists(cache_file):
        pack = torch.load(cache_file, map_location="cpu")
        input_ids = pack["input_ids"]
        attention_mask = pack["attention_mask"]
        return _make_loader_from_tensors(input_ids, attention_mask, batch_size=batch_size)

    # 2) 加载数据集
    cfg = _dataset_config_by_name(name)

    if cfg["type"] == "text":
        dataset = load_dataset("text", data_files={"train": cfg["path"]})["train"]
    elif cfg["type"] == "parquet":
        dataset = load_dataset("parquet", data_files={"train": cfg["path"]})["train"]
    else:
        raise ValueError(f"Unsupported dataset type: {cfg['type']}")

    L = len(dataset)

    # ✅ nsamples=0 取全量，否则随机采样
    if nsamples == 0:
        indices = list(range(L))
    else:
        n = min(nsamples, L)
        indices = np.random.choice(L, size=n, replace=False).astype(int).tolist()

    samples = []
    for idx in indices:
        sample = dataset[idx]
        text = sample[cfg["text_field"]]
        if isinstance(text, list):
            text = " ".join(map(str, text))
        samples.append(str(text).strip())

    # 3) tokenization
    calib_dataset = CalibrationDataset(samples=samples, tokenizer=tokenizer, seqlen=seqlen)
    tmp_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    all_input_ids = []
    all_attention_mask = []
    for batch in tmp_loader:
        all_input_ids.append(batch["input_ids"].cpu())
        all_attention_mask.append(batch["attention_mask"].cpu())

    input_ids = torch.cat(all_input_ids, dim=0)            # [N, T]
    attention_mask = torch.cat(all_attention_mask, dim=0)  # [N, T]

    # 4) 写缓存
    torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, cache_file)

    # 5) 返回最终 DataLoader
    return _make_loader_from_tensors(input_ids, attention_mask, batch_size=batch_size)


#============================================#
#              获取测试数据集                #
#============================================#
def get_full_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    # 获取当前脚本文件的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, 'datasets')
    print('============', datasets_dir)

    if 'wikitext2' in name:
        file_path = os.path.join(datasets_dir, 'wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet')
        test_data = pd.read_parquet(file_path)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif 'ptb' in name:
        # 从本地文件加载 PTB 数据集
        ptb_file_path = os.path.join(datasets_dir, 'ptb/ptb.test.txt')
        with open(ptb_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 将每一行作为一个样本
        samples = {'sentence': [line.strip() for line in lines]}
        test_dataset = process_data(samples, tokenizer, seq_len, 'sentence')

    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader




def get_part_test_data(dataset, tokenizer, seq_len, batch_size, num_samples, seed=3):
    # 设置随机种子
    rng = np.random.RandomState(seed)
    
    # 加载数据集
    if dataset == "wikitext2":
        path = Path("./utils/datasets/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet")
        df = pd.read_parquet(path)
        text = "\n".join(df["text"].tolist())
    elif dataset == "ptb":
        path = Path("./utils/datasets/ptb/ptb.test.txt")
        text = path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Tokenization（返回numpy数组）
    encoded = tokenizer.encode(text, add_special_tokens=False)
    tokens = np.array(encoded)  # 转换为numpy数组

    # 分块处理
    block_size = seq_len
    n_blocks = len(tokens) // block_size
    truncated = tokens[:n_blocks * block_size]
    
    # 直接重塑为二维数组（避免列表转换）
    sequences = truncated.reshape(-1, block_size)

    # 随机采样
    if num_samples > len(sequences):
        selected = sequences
    else:
        indices = rng.choice(len(sequences), num_samples, replace=False)
        selected = sequences[indices]

    # 转换为张量（单次转换）
    input_ids = torch.from_numpy(selected).long()

    # 创建DataLoader
    def collate_fn(batch):
        return torch.stack([x[0] for x in batch])

    return DataLoader(
        TensorDataset(input_ids),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # 暂时禁用多进程加载
    )


#=====Arc-e, Arc-c=====#

import json

# 你已有的 collate_fn / CalibrationDataset 不用改，这里单独给 ARC 做一个 dataset（带 labels）


def _normalize_arc_name(name: str) -> str:
    n = (name or "").strip().lower()
    alias = {
        "arc_e": "arc_e",
        "arc-e": "arc_e",
        "arc_easy": "arc_e",
        "arc-easy": "arc_e",
        "easy": "arc_e",
        "arc_easy_train": "arc_e",

        "arc_c": "arc_c",
        "arc-c": "arc_c",
        "arc_challenge": "arc_c",
        "arc-challenge": "arc_c",
        "challenge": "arc_c",
        "arc_challenge_train": "arc_c",
    }
    if n in alias:
        return alias[n]
    if "easy" in n:
        return "arc_e"
    if "chall" in n or "challenge" in n:
        return "arc_c"
    raise ValueError(f"Unknown ARC name: {name} (use arc_e / arc_c)")


def _arc_parquet_path(ds_key: str) -> str:
    if ds_key == "arc_c":
        return "./utils/datasets/arc/challenge/train-00000-of-00001.parquet"
    return "./utils/datasets/arc/easy/train-00000-of-00001.parquet"


def _maybe_json(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def _extract_arc_parts(sample: dict):
    """
    尽量从一条 ARC 样本里抽出 stem / choices(list[(label,text)]) / answerKey
    兼容 dict/list/json-string 的情况。
    """
    # answer
    ans = sample.get("answerKey", None)
    if ans is None:
        ans = sample.get("answer", None)
    if ans is None:
        ans = sample.get("label", None)
    ans = _maybe_json(ans)
    ans = "" if ans is None else str(ans).strip()

    # question & choices
    q = _maybe_json(sample.get("question", None))
    stem = ""
    choices = None

    if isinstance(q, dict):
        stem = q.get("stem", "") or q.get("question", "") or q.get("prompt", "") or ""
        choices = q.get("choices", None)
    elif isinstance(q, str):
        stem = q
        choices = sample.get("choices", None)
    else:
        stem = sample.get("stem", "") or sample.get("question_stem", "") or sample.get("prompt", "") or ""
        choices = sample.get("choices", None)

    stem = str(stem).strip()
    choices = _maybe_json(choices)

    rendered = []
    if isinstance(choices, dict):
        labels = choices.get("label", []) or choices.get("labels", []) or []
        texts = choices.get("text", []) or choices.get("texts", []) or choices.get("options", []) or []
        for lab, txt in zip(labels, texts):
            rendered.append((str(lab).strip(), str(txt).strip()))
    elif isinstance(choices, list):
        for c in choices:
            c = _maybe_json(c)
            if isinstance(c, dict):
                lab = str(c.get("label", "") or c.get("key", "") or "").strip()
                txt = str(c.get("text", "") or c.get("value", "") or c.get("option", "") or "").strip()
                rendered.append((lab, txt))
            else:
                rendered.append(("", str(c).strip()))

    # 如果 answerKey 是 "1~5" 这种数字，映射到 A~E（尽量兼容）
    if ans.isdigit():
        m = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        ans = m.get(ans, ans)

    return stem, rendered, ans


def _format_arc_prompt(stem: str, choices: list) -> str:
    lines = []
    lines.append(f"Question: {stem}")
    if choices:
        lines.append("Choices:")
        for lab, txt in choices:
            if not txt:
                continue
            if lab:
                lines.append(f"({lab}) {txt}")
            else:
                lines.append(f"- {txt}")
    # 注意：让模型只学习补全 Answer 后面的 token
    lines.append("Answer:")
    return "\n".join(lines)


class ArcSFTDataset(Dataset):
    """
    返回 {"input_ids","attention_mask","labels"}
    labels 只在 Answer 后面计算 loss，其它部分设为 -100
    """
    def __init__(self, prompts, answers, tokenizer, seqlen: int):
        self.prompts = prompts
        self.answers = answers
        self.tokenizer = tokenizer
        self.seqlen = seqlen

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        ans = self.answers[idx]

        # full = prompt + " " + ans
        full_text = prompt + " " + str(ans).strip()

        # 1) encode prompt（为了找到 Answer 起点）
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            padding=False,
            max_length=self.seqlen,
            return_tensors="pt",
        )
        prompt_ids = prompt_enc["input_ids"][0]

        # 若 tokenizer 给 prompt 末尾加了 eos，把它去掉，否则 prompt_len 会偏 1
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and prompt_ids.numel() > 0 and int(prompt_ids[-1]) == int(eos_id):
            prompt_len = int(prompt_ids.numel() - 1)
        else:
            prompt_len = int(prompt_ids.numel())

        # 2) encode full（pad 到 max_length）
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.seqlen,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"][0]
        attention_mask = full_enc["attention_mask"][0]

        # 3) labels：只在 answer 部分有监督
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _make_loader_from_tensors_with_labels(input_ids: torch.Tensor,
                                         attention_mask: torch.Tensor,
                                         labels: torch.Tensor,
                                         batch_size: int) -> DataLoader:
    ds = TensorDataset(input_ids, attention_mask, labels)

    def _collate(tuples):
        ii = torch.stack([t[0] for t in tuples])
        am = torch.stack([t[1] for t in tuples])
        lb = torch.stack([t[2] for t in tuples])
        return {"input_ids": ii, "attention_mask": am, "labels": lb}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate, pin_memory=True)


def get_calib_train_data_acc(
    name: str,               # "arc_e" / "arc_c"
    tokenizer,
    nsamples: int,
    seqlen: int = 512,       # ARC 建议 256/512/1024，2048 太浪费
    seed: int = 2,
    batch_size: int = 1,
    cache_dir: str = "cache",
) -> DataLoader:
    """
    本地 parquet 的 ARC-E / ARC-C 训练集，返回 DataLoader(dict batch)：
      {"input_ids":[B,T], "attention_mask":[B,T], "labels":[B,T]}

    nsamples:
      - >0: 随机不放回抽 nsamples 条
      - 0 : 取全量
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(cache_dir, exist_ok=True)

    ds_key = _normalize_arc_name(name)
    ns_key = "all" if nsamples == 0 else str(nsamples)
    cache_file = os.path.join(cache_dir, f"{ds_key}_{ns_key}_{seqlen}_{seed}_{batch_size}_sft.pt")

    # 1) cache
    if os.path.exists(cache_file):
        pack = torch.load(cache_file, map_location="cpu")
        return _make_loader_from_tensors_with_labels(
            pack["input_ids"], pack["attention_mask"], pack["labels"], batch_size=batch_size
        )

    # 2) load parquet
    path = _arc_parquet_path(ds_key)
    dataset = load_dataset("parquet", data_files={"train": path})["train"]
    L = len(dataset)
    if L == 0:
        raise RuntimeError(f"Empty ARC parquet: {path}")

    if nsamples == 0:
        indices = list(range(L))
    else:
        n = min(int(nsamples), L)
        indices = np.random.choice(L, size=n, replace=False).astype(int).tolist()

    prompts, answers = [], []
    for idx in indices:
        s = dataset[int(idx)]
        stem, choices, ans = _extract_arc_parts(s)
        if not stem or not ans:
            continue
        prompt = _format_arc_prompt(stem, choices)
        prompts.append(prompt)
        answers.append(ans)

    if len(prompts) == 0:
        raise RuntimeError("No valid ARC samples extracted (check parquet schema/fields).")

    # 3) tokenize to tensors（一次性做完并缓存）
    tmp_ds = ArcSFTDataset(prompts, answers, tokenizer, seqlen=seqlen)
    tmp_loader = DataLoader(tmp_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    all_i, all_m, all_l = [], [], []
    for b in tmp_loader:
        all_i.append(b["input_ids"].cpu())
        all_m.append(b["attention_mask"].cpu())
        all_l.append(b["labels"].cpu())

    input_ids = torch.cat(all_i, dim=0)
    attention_mask = torch.cat(all_m, dim=0)
    labels = torch.cat(all_l, dim=0)

    torch.save({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}, cache_file)

    return _make_loader_from_tensors_with_labels(input_ids, attention_mask, labels, batch_size=batch_size)
