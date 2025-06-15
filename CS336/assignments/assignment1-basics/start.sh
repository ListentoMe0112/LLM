python train.py \
  --train_data /Users/yeeboxie/Documents/Learn/LLM/CS336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.bin \
  --val_data /Users/yeeboxie/Documents/Learn/LLM/CS336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.bin \
  --out_dir ./models \
  --log_interval 1 \
  --context_length 1024 \
  --device mps
