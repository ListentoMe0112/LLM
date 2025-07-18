import argparse
import json
import numpy as np
from tqdm import tqdm
import os
import tests.common as common
import tests.adapters as ada
import tiktoken

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

def main():
    parser = argparse.ArgumentParser(description='Convert text file to binary token IDs')
    parser.add_argument('--input', type=str, required=True, help='Input text file path')
    parser.add_argument('--output', type=str, required=True, help='Output binary file path')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary JSON file')
    parser.add_argument('--merges', type=str, required=True, help='Merges TXT file')
    args = parser.parse_args()

    tokenizer = get_tokenizer_from_vocab_merges_path(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    
    # Process text file in chunks
    token_ids = []
    
    with open(args.input, 'r', encoding='utf-8') as f:
        token_ids = tokenizer.encode(f.read(),max_workers=16)
    # Convert to numpy array and save as binary
    arr = np.array(token_ids, dtype=np.int32)
    arr.tofile(args.output)
    print(f"Saved {len(token_ids)} tokens to {args.output}")

if __name__ == "__main__":
    main()


