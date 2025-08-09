# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 1600 6400 48 25
# 2.7B 2560 10240 32 32
uv run nsys profile --python-backtrace=cuda -o result_origin python benchmark_for_mem.py --d_model=2560 --d_ff=10240 --num_layers=32 --num_heads=32 --run_backward=True --warm_up=3 --iteration=20

