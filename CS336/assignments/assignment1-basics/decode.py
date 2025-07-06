from cs336_basics import utils
import torch

import argparse
import json
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import os
import tests.common as common
import tests.adapters as ada
import re  # 导入正则表达式模块

# 初始化模型
model = utils.TransformerLanguageModel(
   vocab_size=50257,
   d_model=768,
   num_heads=12,
   theta=10000.0,
   max_seq_len=1024,
   d_ff=3072,
   num_layers=12,
   device="cpu" 
)


# 加载模型权重
checkpoint = torch.load("./models/checkpoint_1000.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # 设置为评估模式

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in common.gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return ada.get_tokenizer(vocab, merges, special_tokens)


tokenizer = get_tokenizer_from_vocab_merges_path("tests/fixtures/gpt2_vocab.json", "tests/fixtures/gpt2_merges.txt", special_tokens=["<|endoftext|>"])

# 生成文本
prompt_text = "Xiaoyi Zhang loves eat shit"
prompt_encoded = tokenizer.encode(prompt_text)
# 解码并清理文本
decoded_text = tokenizer.decode(prompt_encoded)
print(decoded_text)

# 将编码后的提示转换为张量
prompt_tensor = torch.tensor([prompt_encoded], device="cpu")

# 生成文本
generated = model.generate(
    prompt=prompt_tensor,
    max_new_tokens=1024,
    end_token_id=tokenizer.encode("<|endoftext|>", max_workers=1)
)

# 将张量转换为列表
generated_ids = generated[0].tolist()

# 解码并清理生成的文本
decoded_text = tokenizer.decode(generated_ids)
print(decoded_text)

