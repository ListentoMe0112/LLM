from tests import adapters
import pickle
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input Data")
    parser.add_argument('-i', '--input_path', type=str, help='path to input', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='path to output', required=True)
    parser.add_argument('-n', '--number_of_token', type=int, help='maximum vocabulary size', required=True)
    args = parser.parse_args()

    vocab, merges = adapters.run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.number_of_token,
        special_tokens=["<|endoftext|>"],
    )

    print("Process Finished!")
    with open(args.output_path, 'wb') as f:
        pickle.dump(vocab, f)
    
