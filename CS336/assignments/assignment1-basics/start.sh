nohup python -u  train.py \
  --train_data ./data/TinyStoriesV2-GPT4-train.bin \
  --val_data ./data/TinyStoriesV2-GPT4-valid.bin \
  --out_dir ./models \
  --log_interval 1 \
  --device mps > training.log 2>&1 &
