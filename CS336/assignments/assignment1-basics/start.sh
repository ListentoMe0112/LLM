nohup python -u train.py \
  --train_data /root/LLM/CS336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.bin \
  --val_data /root/LLM/CS336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.bin \
  --out_dir ./models \
  --log_interval 1 \
  --device cuda > training.log 2>&1 &
