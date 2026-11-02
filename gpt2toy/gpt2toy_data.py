"""数据准备：规则分词、因果语言建模的滑动窗口，以及批次装载。"""

import re

import torch
from torch.utils.data import DataLoader, Dataset


class SimpleTokenizerV2:
    """规则分词器（Rule-based Tokenizer）：使用正则拆分词语和标点，并查微型词表。

    通过词元与整数 ID 的双向映射实现编码/解码；未知词映射到 <|unk|>。
    分词会丢弃空白，解码不保证完整恢复原文的空白布局。
    """

    def __init__(self, vocab):
        """根据给定词表建立 Token -> ID 与 ID -> Token 两张查找表。"""
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        """分词（Tokenization）后把字符串编码为 Token ID 列表，保留可识别的标点。"""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """把 ID 列表还原为词元文本，并消除部分标点前多余的连接空格。"""
        text = " ".join([self.int_to_str.get(i, "<|unk|>") for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


class GPTDatasetV1(Dataset):
    """因果语言建模（Causal Language Modeling，CLM）的滑动窗口数据集。

    每个样本包含长度 T 的输入，以及向右偏移一位的目标；模型在各位置
    学习下一个 Token 预测（Next-Token Prediction，NTP）。
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """按 stride 滑动窗口，仅保留同时具有完整输入与右移目标的片段。"""
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词 [Tokenize]
        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口 [Sliding Window] 将整本书切分为最大长度 [max_length] 的重叠序列 [Sequences]
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回可用于训练的完整窗口数量。"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """返回一个 (input_ids, target_ids) 样本，两者形状均为 [T]。"""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """用 DataLoader 将窗口样本组成批次，返回两组 [B,T] 的输入/目标张量。

    shuffle 控制样本打乱，drop_last 控制是否丢弃不足一个批次的尾部样本。
    """
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader
