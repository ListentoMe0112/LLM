import argparse
from cs336_data.utils import run_extract_text_from_html_bytes_implementation

def main(input_path: str, output_path: str):
    with open(input_path, "rb") as f:
        moby_bytes = f.read()
    extract = run_extract_text_from_html_bytes_implementation(moby_bytes)
    with open(output_path) as f:
        f.write(extract)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./data/input.warc", help='input path!')
    parser.add_argument('--output', type=str, default="./data/output.wet", help='output path!')
    args = parser.parse_args()
    main(args.input, args.output)