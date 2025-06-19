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
checkpoint = torch.load("./models/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # 设置为评估模式

with open("./gpt2_tokenizer/vocab.json", 'r') as f:
    vocab = json.load(f)
    
with open("./gpt2_tokenizer/merges.txt", 'r', encoding='utf-8') as f:
    merges = f.read().split('\n')[1:-1]  # 跳过第一行和最后一行
    merges = [tuple(merge.split()) for merge in merges]
    
bpe = BPE(
    vocab=vocab,
    merges=merges,
    unk_token="<|endoftext|>"
)

tokenizer = Tokenizer(bpe)
    
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.post_processor = TemplateProcessing(
    single="<|endoftext|> $A <|endoftext|>",
    special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))]
)

# 自定义函数：去除Ġ字符并处理空格
def clean_decoded_text(text):
    # 去除Ġ字符
    cleaned = text.replace('Ġ', ' ')
    # 合并多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # 去除开头和结尾空格
    cleaned = cleaned.strip()
    return cleaned

# 生成文本
prompt_text = "Write a short story about Tom and Lily's adventure"
prompt_encoded = tokenizer.encode(prompt_text)
# 解码并清理文本
decoded_text = tokenizer.decode(prompt_encoded.ids)
cleaned_text = clean_decoded_text(decoded_text)
print(cleaned_text)

# 将编码后的提示转换为张量
prompt_tensor = torch.tensor([prompt_encoded.ids], device="cpu")

# 生成文本
generated = model.generate(
    prompt=prompt_tensor,
    max_new_tokens=100,
    #temperature=0.7,  # 降低温度减少随机性
    top_p=0.95,       # 提高top-p增加连贯性
    end_token_id=tokenizer.token_to_id("<|endoftext|>")
)

# 将张量转换为列表
generated_ids = generated[0].tolist()

decoded_text = tokenizer.decode(generated_ids)

print(decoded_text)
