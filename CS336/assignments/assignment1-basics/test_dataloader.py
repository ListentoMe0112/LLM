from cs336_basics.utils import (
    TransformerLanguageModel,
    AdamW,
    get_batch,
    cross_entropy,
    gradient_clipping,
    learning_rate_schedule,
    save_checkpoint,
    load_checkpoint
)

import numpy as np
import os
import json
import tests.common as common
import tests.adapters as ada

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
train_data_path = "./data/TinyStoriesV2-GPT4-train.bin" 

train_data = np.memmap(train_data_path, dtype=np.int32, mode='r')


x, y = get_batch(
    dataset=train_data,
    batch_size=1,
    context_length=1024,
    device="mps"
)

generated_ids = x[0].tolist()

print(tokenizer.decode(generated_ids))


