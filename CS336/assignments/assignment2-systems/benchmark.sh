# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 1600 6400 48 25
# 2.7B 2560 10240 32 32
# uv run nsys profile -o result python benchmark.py
uv run nsys profile --python-backtrace=cuda -o result python benchmark.py --d_model=768 --d_ff=3072 --num_layers=12 --num_heads=12 --run_backward=True --warm_up=3 --iteration=5
# too large for single v100, try a100 later
# python benchmark.py --d_model=1024 --d_ff=4096 --num_layers=24 --num_heads=16 --run_backward=True --warm_up=1 --iteration=10
# python benchmark.py --d_model=1280 --d_ff=5120 --num_layers=36 --num_heads=20 --run_backward=True --warm_up=1  
# python benchmark.py --d_model=1600 --d_ff=6400 --num_layers=48 --num_heads=25 --run_backward=True --warm_up=1  
# python benchmark.py --d_model=2560 --d_ff=10240 --num_layers=32 --num_heads=32 --run_backward=True --warm_up=1  
