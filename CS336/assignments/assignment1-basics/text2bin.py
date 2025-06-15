import argparse
import json
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Convert text file to binary token IDs')
    parser.add_argument('--input', type=str, required=True, help='Input text file path')
    parser.add_argument('--output', type=str, required=True, help='Output binary file path')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary JSON file')
    parser.add_argument('--merges', type=str, required=True, help='Merges TXT file')
    args = parser.parse_args()

    # 修复方法：使用正确的BPE初始化方式
    # 1. 加载词汇表
    with open(args.vocab, 'r') as f:
        vocab = json.load(f)
    
    # 2. 加载合并规则
    with open(args.merges, 'r', encoding='utf-8') as f:
        merges = f.read().split('\n')[1:-1]  # 跳过第一行和最后一行
        merges = [tuple(merge.split()) for merge in merges]
    
    # 3. 创建BPE模型
    bpe = BPE(
        vocab=vocab,
        merges=merges,
        unk_token="<|endoftext|>"
    )

    # 4. 创建tokenizer
    tokenizer = Tokenizer(bpe)
    
    # 配置tokenizer
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.post_processor = TemplateProcessing(
        single="<|endoftext|> $A <|endoftext|>",
        special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))]
    )


    # Process text file in chunks
    token_ids = []
    chunk_size = 100000  # Process 100KB chunks for memory efficiency
    
    with open(args.input, 'r', encoding='utf-8') as f:
        with tqdm(desc="Tokenizing", unit=" tokens") as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                # Tokenize and add to results
                encoding = tokenizer.encode(chunk)
                token_ids.extend(encoding.ids)
                pbar.update(len(encoding.ids))

    # Convert to numpy array and save as binary
    arr = np.array(token_ids, dtype=np.int32)
    arr.tofile(args.output)
    print(f"Saved {len(token_ids)} tokens to {args.output}")

if __name__ == "__main__":
    main()

