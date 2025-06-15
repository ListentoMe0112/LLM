python text2bin.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinyStoriesV2-GPT4-train.bin \
  --vocab gpt2_tokenizer/vocab.json \
  --merges gpt2_tokenizer/merges.txt
python text2bin.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/TinyStoriesV2-GPT4-valid.bin \
  --vocab gpt2_tokenizer/vocab.json \
  --merges gpt2_tokenizer/merges.txt
