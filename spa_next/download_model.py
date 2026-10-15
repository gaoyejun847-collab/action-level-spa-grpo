"""Download the explicitly selected Qwen model using the existing ModelScope client."""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    from modelscope import snapshot_download
    print(snapshot_download("Qwen/Qwen3-4B-Instruct-2507", local_dir=args.directory))
