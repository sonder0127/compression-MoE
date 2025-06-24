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
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
# # 获取代理数据集
# def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
#     import random
#     random.seed(seed)
    
#     mathqa_ds = load_dataset(
#     "parquet",
#     data_files={
#         "train": "./utils/datasets/mathqa/data/train-00000-of-00001.parquet"
#     }
#     )
#     arc_easy_ds = load_dataset(
#     "parquet",
#     data_files={
#         "train": "./utils/datasets/arc/easy/train-00000-of-00001.parquet"
#     }
#     )
#     arc_challenge_ds = load_dataset(
#     "parquet",
#     data_files={
#         "train": "./utils/datasets/arc/challenge/train-00000-of-00001.parquet"
#     }
#     )
#     openbookqa_ds = load_dataset(
#     "arrow",
#     data_files={
#         "train": "./utils/datasets/openbookqa/train/data-00000-of-00001.arrow"
#     }
#     )
#     ptb_ds = load_dataset(
#     "txt",
#     data_files={
#         "train": "./utils/datasets/ptb.train.txt"
#     }
#     )
#     wikitext_ds = load_dataset(
#     "parquet",
#     data_files={
#         "train": "./utils/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
#     }
#     )

class CalibrationDataset(Dataset):
    def __init__(self, samples, tokenizer, seqlen):
        self.samples = samples
        self.tokenizer = tokenizer
        self.seqlen = seqlen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 动态处理文本长度
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.seqlen,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': encoding.input_ids[0],
            'attention_mask': encoding.attention_mask[0]
        }

def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch])
    }

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1):
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cache_file = (
        f"cache/{name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    datasets = [
        {
            "type": "text",
            "path": "./utils/datasets/ptb/ptb.train.txt",
            "text_field": "text"
        },
        {
            "type": "parquet",
            "path": "./utils/datasets/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet",
            "text_field": "text"
        }
    ]

    collected_samples = []
    samples_per_dataset = nsamples // 2

    for ds_config in datasets:
        try:
            # 1. 加载数据集
            if ds_config["type"] == "text":
                dataset = load_dataset("text", data_files={"train": ds_config["path"]})["train"]
            else:
                dataset = load_dataset("parquet", data_files={"train": ds_config["path"]})["train"]
            # 2. 类型转换关键步骤
            dataset_length = len(dataset)
            indices = np.random.choice(dataset_length,
                                     size=min(samples_per_dataset, dataset_length),
                                     replace=False)
            # 将 numpy.int64 转换为 Python int
            indices = indices.astype(int).tolist()  # 双重转换确保类型正确
            # 3. 正确获取文本数据
            for idx in indices:
                # 确保索引是原生Python整数类型
                assert isinstance(idx, int), f"索引类型错误: {type(idx)} 应为 int"

                # 获取文本字段
                sample = dataset[idx]
                text = sample[ds_config["text_field"]]

                # 处理可能的列表格式
                if isinstance(text, list):
                    text = " ".join(map(str, text))
                collected_samples.append(str(text).strip())

            print(f"成功从 {os.path.basename(ds_config['path'])} 采集 {len(indices)} 个样本")

        except Exception as e:
            print(f"加载数据集失败: {str(e)}")
            continue
    # 创建数据集
    calib_dataset = CalibrationDataset(
        samples=collected_samples[:nsamples],
        tokenizer=tokenizer,
        seqlen=seqlen
    )
    # 创建数据加载器
    dataloader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    torch.save(dataloader, cache_file)

    return dataloader
    
def get_calib_train_data_old(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1):
    import random
    random.seed(seed)
    print('datasets', name)
    cache_file = (
        f"cache/{name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )
    nsamples += 1 
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        print("ptb...")
        # 从本地文件加载 ptb 数据集
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, 'datasets')
        file_path = os.path.join(datasets_dir, 'ptb/ptb.train.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            traindata = f.readlines()
        tot_text = "\n\n".join(traindata)
    
    elif name == "wikitext2":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        datasets_dir = os.path.join(script_dir, 'datasets')
        file_path = os.path.join(datasets_dir, 'wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet')
        traindata = pd.read_parquet(file_path)
        tot_text = "\n\n".join(traindata["text"])
        
    else:
        raise NotImplementedError
    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    torch.save(traindataset, cache_file)
    return traindataset

# 获取全部数据集
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

# 获取数据集特定个数的样本
def get_part_test_data(name, tokenizer, seq_len=2048, batch_size=4, num_samples=None, seed=3):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name=None, num_samples=None):
        if num_samples is not None:
            # 创建独立的随机生成器以避免影响全局状态
            if seed is not None:
                rng = random.Random(seed)
            else:
                rng = random.Random()  # 使用默认种子（基于系统时间）

            # 打乱样本的副本
            samples_copy = samples.copy()
            rng.shuffle(samples_copy)
            samples = samples_copy[:num_samples]
        if num_samples is not None:
            samples = samples[:num_samples]

        if isinstance(samples[0], str):
            # 如果 samples 是字符串列表，直接使用字符串
            texts = "\n\n".join(samples)
        else:
            # 如果 samples 是字典列表，使用 field_name 访问字典值
            texts = "\n\n".join([str(sample[field_name]) for sample in samples])

        test_ids = tokenizer(texts, return_tensors='pt').input_ids[0]

        test_ids_batch = []
        nsamples = (test_ids.numel() - 1) // seq_len  # Adjusted to account for the shift in perplexity calculation

        if nsamples == 0:
            print("Warning: No full sequences found after tokenization.")
            return IndexDataset(tensors=torch.tensor([]))  # Return an empty dataset if no full sequences are available

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)

        if not test_ids_batch:  # Ensure there's at least one batch to stack
            print("Warning: No batches created from the provided data.")
            return IndexDataset(tensors=torch.tensor([]))

        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, 'datasets')

    if 'wikitext2' in name:
        file_path = os.path.join(datasets_dir, 'wikitext/wikitext-2-v1/test-00000-of-00001.parquet')
        test_data = pd.read_parquet(file_path).to_dict('records')  # Convert DataFrame to list of dictionaries for consistency
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text', num_samples)
    elif 'ptb' in name:
        # 假设本地的 ptb 数据集文件名为 ptb_test.parquet
        file_path = os.path.join(datasets_dir, 'ptb/ptb.test.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            test_texts = f.readlines()
        test_dataset = process_data(test_texts, tokenizer, seq_len, num_samples=num_samples)
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text', num_samples)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def get_part_test_data_old(name, tokenizer, seq_len=2048, batch_size=4, num_samples=None, seed=3):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name=None, num_samples=None):
        if num_samples is not None:
            # 创建独立的随机生成器以避免影响全局状态
            if seed is not None:
                rng = random.Random(seed)
            else:
                rng = random.Random()  # 使用默认种子（基于系统时间）

            # 打乱样本的副本
            samples_copy = samples.copy()
            rng.shuffle(samples_copy)
            samples = samples_copy[:num_samples]
        if num_samples is not None:
            samples = samples[:num_samples]

        if isinstance(samples[0], str):
            # 如果 samples 是字符串列表，直接使用字符串
            texts = "\n\n".join(samples)
        else:
            # 如果 samples 是字典列表，使用 field_name 访问字典值
            texts = "\n\n".join([str(sample[field_name]) for sample in samples])

        test_ids = tokenizer(texts, return_tensors='pt').input_ids[0]

        test_ids_batch = []
        nsamples = (test_ids.numel() - 1) // seq_len  # Adjusted to account for the shift in perplexity calculation

        if nsamples == 0:
            print("Warning: No full sequences found after tokenization.")
            return IndexDataset(tensors=torch.tensor([]))  # Return an empty dataset if no full sequences are available

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)

        if not test_ids_batch:  # Ensure there's at least one batch to stack
            print("Warning: No batches created from the provided data.")
            return IndexDataset(tensors=torch.tensor([]))

        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, 'datasets')

    if 'wikitext2' in name:
        file_path = os.path.join(datasets_dir, 'wikitext/wikitext-2-v1/test-00000-of-00001.parquet')
        test_data = pd.read_parquet(file_path).to_dict('records')  # Convert DataFrame to list of dictionaries for consistency
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text', num_samples)
    elif 'ptb' in name:
        # 假设本地的 ptb 数据集文件名为 ptb_test.parquet
        file_path = os.path.join(datasets_dir, 'ptb/ptb.test.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            test_texts = f.readlines()
        test_dataset = process_data(test_texts, tokenizer, seq_len, num_samples=num_samples)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# 禁用tokenizers并行处理（必须在导入tokenizers之前设置）
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_part_test_data(dataset, tokenizer, seq_len, batch_size, num_samples, seed=3):
    # 设置随机种子
    rng = np.random.RandomState(seed)
    
    # 加载数据集
    if dataset == "wikitext2":
        path = Path("./utils/datasets/wikitext/wikitext-2-v1/test-00000-of-00001.parquet")
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


def get_test_data_local_part_datasets(name, tokenizer, seq_len=2048, batch_size=4, num_samples=None):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name, num_samples=None):
        if num_samples is not None:
            samples = samples[:num_samples]
        
        # Tokenize all samples at once and then split into sequences of seq_len
        texts = "\n\n".join([str(sample[field_name]) for sample in samples])
        test_ids = tokenizer(texts, return_tensors='pt').input_ids[0]
        
        test_ids_batch = []
        nsamples = (test_ids.numel() - 1) // seq_len  # Adjusted to account for the shift in perplexity calculation
        
        if nsamples == 0:
            print("Warning: No full sequences found after tokenization.")
            return IndexDataset(tensors=torch.tensor([]))  # Return an empty dataset if no full sequences are available

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        
        if not test_ids_batch:  # Ensure there's at least one batch to stack
            print("Warning: No batches created from the provided data.")
            return IndexDataset(tensors=torch.tensor([]))
        
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(script_dir, 'datasets')
    print('============', datasets_dir)

    if 'wikitext2' in name:
        file_path = os.path.join(datasets_dir, 'wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet')
        test_data = pd.read_parquet(file_path).to_dict('records')  # Convert DataFrame to list of dictionaries for consistency
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text', num_samples)
    elif 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence', num_samples)
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text', num_samples)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader