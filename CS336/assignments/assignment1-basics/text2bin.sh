python text2bin.py \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/TinyStoriesV2-GPT4-train.bin \
  --vocab tests/fixtures/gpt2_vocab.json \
  --merges tests/fixtures/gpt2_merges.txt
python text2bin.py \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/TinyStoriesV2-GPT4-valid.bin \
  --vocab tests/fixtures/gpt2_vocab.json \
  --merges tests/fixtures/gpt2_merges.txt
