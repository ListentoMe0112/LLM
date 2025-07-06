import cs336_basics
import argparse
import tests

def main():
    parser = argparse.ArgumentParser(description='Convert text file to binary token IDs')
    parser.add_argument('--input', type=str, required=True, help='Input text file path')
    parser.add_argument('--output', type=str, required=True, help='Output binary file path')
    args = parser.parse_args()
    vocab, merges = tests.run_train_bpe(
        input_path = args.input, 
        vocab_size = 10000,
        special_tokens=["<|endoftext|>"],
    )



