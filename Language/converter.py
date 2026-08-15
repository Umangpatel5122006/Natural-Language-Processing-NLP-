import pandas as pd
import argparse
import os
import sys
import tempfile


def read_lines_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stream_hf_dataset(limit=None):
    """Stream the ai4bharat/IndicCorpV2 gujarati split and yield text strings.

    Attempts to detect a suitable text field in each example. Requires the
    `datasets` library to be installed (pip install datasets).
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        print(
            "The 'datasets' library is required to download from Hugging Face.\n"
            "Install it with: pip install datasets",
            file=sys.stderr,
        )
        raise

    ds = load_dataset(
        "ai4bharat/IndicCorpV2",
        "indiccorp_v2",
        split="guj_Gujr",
        streaming=True,
    )

    count = 0
    for ex in ds:
        # ex is typically a dict-like record. Find the first string field.
        if isinstance(ex, dict):
            text_val = None
            # Common candidate keys
            candidates = ["text", "sentence", "content", "translation", "line"]
            for k in candidates:
                if k in ex and isinstance(ex[k], str):
                    text_val = ex[k]
                    break
            if text_val is None:
                # fallback: pick first string-valued field
                for k, v in ex.items():
                    if isinstance(v, str):
                        text_val = v
                        break
                    # sometimes values are list of strings
                    if isinstance(v, (list, tuple)) and v and all(isinstance(i, str) for i in v):
                        text_val = " ".join(v)
                        break
            if text_val is None:
                # unable to extract a string from this example; skip
                continue
            yield text_val.strip()
        else:
            # example is not a dict (unexpected) — try converting to str
            yield str(ex).strip()

        count += 1
        if limit is not None and count >= limit:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Convert a local TXT file or the IndicCorpV2 Gujarati split to Parquet."
    )
    parser.add_argument("--input", "-i", help="Path to local input TXT file")
    parser.add_argument(
        "--output", "-o", default="gujarati.parquet", help="Output parquet file path"
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help=(
            "If set, try downloading the Hugging Face IndicCorpV2 gujarati split when local input is not provided. "
            "If not set and no local file is found, the script will exit with an error."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: limit number of records to process (useful for testing)",
    )

    args = parser.parse_args()

    input_path = None

    # Priority: explicit local input if it exists
    if args.input:
        if os.path.exists(args.input):
            input_path = args.input
        else:
            print(f"Input file '{args.input}' not found.", file=sys.stderr)

    texts = None

    if input_path is not None:
        texts = read_lines_from_file(input_path)
    else:
        # No local input — use HF dataset if allowed
        if args.use_hf:
            try:
                print("Streaming IndicCorpV2 (gujarati) from Hugging Face...")
                generator = stream_hf_dataset(limit=args.limit)
                # Collect into list (convertible to DataFrame)
                texts = [t for t in generator]
            except Exception as e:
                print(f"Failed to load Hugging Face dataset: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            default_input = r"E:\STUDY\NLP\Language\gu.txt"
            if os.path.exists(default_input):
                input_path = default_input
                texts = read_lines_from_file(input_path)
                print(f"Using default local input: {default_input}")
            else:
                parser.print_help()
                print(
                    "\nError: No input file found. Either provide --input or use --use-hf to download the dataset.",
                    file=sys.stderr,
                )
                sys.exit(1)

    # Convert to DataFrame and write Parquet
    df = pd.DataFrame({"text": texts})
    df.to_parquet(args.output, index=False)
    print(f"Converted {len(df)} records to '{args.output}'")
    print(df.head())


if __name__ == "__main__":
    main()
