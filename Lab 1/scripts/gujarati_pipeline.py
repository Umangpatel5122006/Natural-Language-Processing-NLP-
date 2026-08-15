r"""
Gujarati corpus tokenization pipeline (streaming, memory-efficient).

Input   : a UTF-8 text file (one paragraph per line) -- default is
          the IndicCorpV2 Gujarati dump at
          E:\STUDY\NLP\Language\gu.txt

Outputs :
  - outputs/tokenized_sentences.txt   one tokenized sentence per line
  - outputs/tokenized_sentences_with_source.txt  JSONL with sentence and token lists
  - outputs/tokenized_sentences.parquet  same rows, columnar + snappy
  - outputs/corpus_stats.json        the six statistics required by
                                     Assignment 1 (d.i-vi)

Pipeline
  1. sentence_tokenize(text)  -> list[str]
  2. word_tokenize(sentence)  -> list[str]
        protects URLs, e-mails, dates, times, decimals from being split
  3. writes each sentence as "<tok1> <tok2> ..." to the .txt
     and accumulates rows for parquet
  4. computes corpus statistics over the tokenized sentences
"""
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter
from typing import List

# Force UTF-8 for Windows consoles and pyarrow text handling.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INPUT = PROJECT_ROOT / "Language" / "gu.txt"
INPUT_PATH = Path(os.environ.get("INPUT_PATH", str(DEFAULT_INPUT))).expanduser()
OUTPUT_TXT = Path(os.environ.get("OUTPUT_TXT", OUTPUT_DIR / "tokenized_sentences.txt")).expanduser()
OUTPUT_DETAILED_TXT = Path(
    os.environ.get(
        "OUTPUT_DETAILED_TXT",
        OUTPUT_DIR / "tokenized_sentences_with_source.txt",
    )
).expanduser()
OUTPUT_PARQUET = Path(
    os.environ.get("OUTPUT_PARQUET", OUTPUT_DIR / "tokenized_sentences.parquet")
).expanduser()
OUTPUT_STATS = Path(os.environ.get("OUTPUT_STATS", OUTPUT_DIR / "corpus_stats.json")).expanduser()
PARAGRAPH_BATCH = 40_000   # paragraphs per streaming batch (keeps RAM low)
PARQUET_BATCH = 100_000    # sentences per parquet write batch
MAX_SENTENCES = 100_000    # Cap the output to exactly 100,000 sentences

# ---------------------------------------------------------------------------
# Tokenizer patterns
# ---------------------------------------------------------------------------
# Gujarati block U+0A80..U+0AFF, Gujarati digits U+0AE6..U+0AEF, ASCII word chars.
_GUJ_BLOCK = r"\u0a80-\u0aff"
_GUJ_DIGIT = r"\u0ae6-\u0aef"
_CHAR_BLOCK = rf"\w{_GUJ_BLOCK}"

# Match protected spans FIRST (longest-first), so URLs aren't torn apart by
# the regex's greedy character class.
_EMAIL_RE = re.compile(
    rf"[{_CHAR_BLOCK}.%+-]+@[{_CHAR_BLOCK}.-]+\.[{_CHAR_BLOCK}]{{2,}}"
)
_URL_RE = re.compile(
    rf"(?:https?://|www\.)[{_CHAR_BLOCK}.-]+\.[{_CHAR_BLOCK}]{{2,}}"
    rf"(?:/[{_CHAR_BLOCK}.%+-]*)*"
)
_DECIMAL_RE = re.compile(rf"[{_GUJ_DIGIT}]+\.[{_GUJ_DIGIT}]+") # for 1.256
_TIME_RE = re.compile(rf"[{_GUJ_DIGIT}]{{1,2}}:[{_GUJ_DIGIT}]{{2}}(?::[{_GUJ_DIGIT}]{{2}})?") # for 12:59:10 
_DATE_RE = re.compile(
    rf"[{_GUJ_DIGIT}]{{1,2}}[./-][{_GUJ_DIGIT}]{{1,2}}[./-][{_GUJ_DIGIT}]{{2,4}}" # for 1/5/2006
)

_PROTECT_ORDER = [_URL_RE, _EMAIL_RE, _DATE_RE, _TIME_RE, _DECIMAL_RE]

GUJARATI_STOPWORDS = {
    "છે",
    "અને",
    "તે",
    "આ",
    "એ",
    "ની",
    "ન",
    "હવે",
    "જ્યારે",
    "હું",
    "મને",
    "મારા",
}

# A sentence ends at one of these terminators, optionally followed by closing
# quotes/brackets, then a hard whitespace boundary.
_SENT_SPLIT_RE = re.compile(
    r"""(?:
        (?<=[.!?\u0964\u0965])   # lookbehind: period / ? / ! / devanagari danda
        ["'\u201d\u2019\)\]\}\u0a8d]*
        \s+
    )""",
    re.VERBOSE,
)

# Word-level separators. We split on whitespace and on every Gujarati/ASCII
# punctuation char that isn't itself a protected span.
_WORD_SEP_RE = re.compile(
    r"""[\s,;:!?\u0964\u0965\u201c\u201d\u2018\u2019"""
    r"""\.\(\)\[\]\{\}\\\/<>@#\$%\^&\*\+=\|`~_""" + r'"\-]+',
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------
def _mask_protected(text: str):
    """Replace protected spans with sentinels, returning (masked_text, originals)."""
    protected: List[str] = []
    masked = text
    for pat in _PROTECT_ORDER:
        def _sub(m, _protected=protected):
            protected.append(m.group(0))
            return f"\x00P{len(protected) - 1}\x00"
        masked = pat.sub(_sub, masked)
    return masked, protected


def _unmask_token(token: str, protected: List[str]) -> str:
    out = token
    for i, original in enumerate(protected):
        sentinel = f"\x00P{i}\x00"
        if sentinel in out:
            out = out.replace(sentinel, original)
    return out


def _unmask_tokens(tokens: List[str], protected: List[str]) -> List[str]:
    return [_unmask_token(t, protected) for t in tokens]


# ---------------------------------------------------------------------------
# Sentence + word tokenizers
# ---------------------------------------------------------------------------
def sentence_tokenize(text: str) -> List[str]:
    if not isinstance(text, str) or not text:
        return []
    text = text.lstrip("\ufeff")
    # Normalize ellipses so they don't accidentally split a sentence.
    text = re.sub(r"\.{2,}|\u2026", " ", text)
    masked, protected = _mask_protected(text)
    parts = _SENT_SPLIT_RE.split(masked)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for i, original in enumerate(protected):
            sentinel = f"\x00P{i}\x00"
            if sentinel in p:
                p = p.replace(sentinel, original)
        out.append(p)
    return out


def word_tokenize(sentence: str, remove_stopwords: bool = False) -> List[str]:
    if not isinstance(sentence, str) or not sentence:
        return []
    sentence = sentence.lstrip("\ufeff")
    masked, protected = _mask_protected(sentence)
    raw = _WORD_SEP_RE.split(masked)
    tokens: List[str] = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue
        tokens.append(tok)
    tokens = _unmask_tokens(tokens, protected)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in GUJARATI_STOPWORDS]
    return tokens


# ---------------------------------------------------------------------------
# Corpus statistics
# ---------------------------------------------------------------------------
def compute_corpus_stats(sentences: List[str]) -> dict:
    if isinstance(sentences, str):
        sentences = [sentences]
    elif not isinstance(sentences, list):
        sentences = list(sentences)
    sentences = [s for s in sentences if isinstance(s, str) and s]
    total_sentences = len(sentences)
    total_words = 0
    total_characters = 0
    total_word_chars = 0
    unique_tokens = set()
    for sentence in sentences:
        toks = word_tokenize(sentence)
        total_words += len(toks)
        total_word_chars += sum(len(tok) for tok in toks)
        total_characters += len(sentence)
        unique_tokens.update(toks)
    average_sentence_length = round(total_words / total_sentences, 4) if total_sentences else 0.0
    average_word_length = round(total_word_chars / total_words, 4) if total_words else 0.0
    type_token_ratio = round(len(unique_tokens) / total_words, 6) if total_words else 0.0
    return {
        "total_sentences": total_sentences,
        "total_words": total_words,
        "total_characters": total_characters,
        "average_sentence_length": average_sentence_length,
        "average_word_length": average_word_length,
        "type_token_ratio": type_token_ratio,
    }


# ---------------------------------------------------------------------------
# Streaming build
# ---------------------------------------------------------------------------
def _build_tokenized_stream(input_path: Path, txt_out: Path, detailed_txt_out: Path, parquet_out: Path):
    schema = pa.schema([
        ("sentence_id", pa.int64()),
        ("sentence", pa.string()),
        ("tokenized_sentence", pa.string()),
        ("word_tokens", pa.string()),
        ("token_count", pa.int64()),
    ])

    sent_id = 0
    total_words = 0
    total_characters = 0
    total_word_chars = 0
    token_counter = Counter()
    sample_rows: List[tuple[str, List[str]]] = []

    # Accumulators grouped
    rows = {"id": [], "sent": [], "tokens": [], "n": []}
    parquet_writer = pq.ParquetWriter(str(parquet_out), schema, compression="snappy")

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as src, \
             open(txt_out, "w", encoding="utf-8", buffering=1 << 20) as dst, \
             open(detailed_txt_out, "w", encoding="utf-8", buffering=1 << 20) as detailed_dst:

            def flush_rows():
                if not rows["id"]:
                    return
                tbl = pa.Table.from_pydict({
                    "sentence_id": rows["id"],
                    "sentence": rows["sent"],
                    "tokenized_sentence": rows["tokens"],
                    "word_tokens": rows["tokens"],
                    "token_count": rows["n"],
                }, schema=schema)
                parquet_writer.write_table(tbl)
                for v in rows.values():
                    v.clear()

            t0 = time.time()
            for raw_line in src:
                paragraph = raw_line.lstrip("\ufeff").strip()
                if not paragraph:
                    continue

                for sentence in sentence_tokenize(paragraph):
                    if MAX_SENTENCES and sent_id >= MAX_SENTENCES:
                        break # Stop processing new sentences

                    toks = word_tokenize(sentence)
                    if not toks:
                        continue

                    sent_id += 1
                    n_toks = len(toks)
                    
                    # Update Metrics
                    total_words += n_toks
                    total_word_chars += sum(len(t) for t in toks)
                    total_characters += len(sentence)
                    token_counter.update(toks) # Counter handles unique tokens automatically

                    tokenized_line = " ".join(toks)

                    # Write to txt and JSONL
                    dst.write(tokenized_line + "\n")
                    detailed_dst.write(json.dumps({
                        "sentence_id": sent_id, "sentence": sentence,
                        "tokenized_sentence": tokenized_line, "word_tokens": toks,
                    }, ensure_ascii=False) + "\n")

                    # Buffer for Parquet
                    rows["id"].append(sent_id)
                    rows["sent"].append(sentence)
                    rows["tokens"].append(tokenized_line)
                    rows["n"].append(n_toks)

                    if len(sample_rows) < 5:
                        sample_rows.append((sentence, toks))

                    if len(rows["id"]) >= PARQUET_BATCH:
                        flush_rows()
                        print(f" sentences={sent_id:>11,}  elapsed={time.time() - t0:7.1f}s", flush=True)

                if MAX_SENTENCES and sent_id >= MAX_SENTENCES:
                    break # Break the outer file-reading loop

            flush_rows()
    finally:
        parquet_writer.close()

    # len(token_counter) gives us the exact count of unique tokens without needing a separate set
    return sent_id, total_words, total_characters, len(token_counter), token_counter, sample_rows, total_word_chars

def main() -> int:
    print(f"Input : {INPUT_PATH}")
    print(
        f"Output: {OUTPUT_TXT} / {OUTPUT_DETAILED_TXT} / {OUTPUT_PARQUET}"
    )
    if MAX_SENTENCES:
        print(f"MAX_SENTENCES cap: {MAX_SENTENCES:,} sentences")

    if not INPUT_PATH.exists():
        print(f"ERROR: input file not found: {INPUT_PATH}", file=sys.stderr)
        return 2

    t0 = time.time()
    sent_n, word_n, char_n, uniq_n, token_counter, sample_rows, total_word_chars = _build_tokenized_stream(
        INPUT_PATH, OUTPUT_TXT, OUTPUT_DETAILED_TXT, OUTPUT_PARQUET
    )
    elapsed = time.time() - t0

    token_stats_csv = OUTPUT_DIR / "token_statistics.csv"
    pd.DataFrame(
        token_counter.most_common(),
        columns=["token", "frequency"],
    ).to_csv(token_stats_csv, index=False, encoding="utf-8")

    stats = {
        "total_sentences": sent_n,
        "total_words": word_n,
        "total_characters": char_n,
        "average_sentence_length": round(word_n / sent_n, 4) if sent_n else 0.0,
        "average_word_length": round(total_word_chars / word_n, 4) if word_n else 0.0,
        "type_token_ratio": round(uniq_n / word_n, 6) if word_n else 0.0,
        "input_path": str(INPUT_PATH),
        "output_txt": str(OUTPUT_TXT),
        "output_detailed_txt": str(OUTPUT_DETAILED_TXT),
        "output_parquet": str(OUTPUT_PARQUET),
        "output_token_stats_csv": str(token_stats_csv),
        "elapsed_seconds": round(elapsed, 2),
        "max_sentences_cap": MAX_SENTENCES, # <--- FIXED
    }
    OUTPUT_STATS.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n========== TOKEN STATISTICS ==========")
    print(f"Total Sentences          : {sent_n}")
    print(f"Total Words              : {word_n}")
    print(f"Total Characters         : {char_n}")
    print(f"Average Sentence Length  : {stats['average_sentence_length']:.2f}")
    print(f"Average Word Length      : {stats['average_word_length']:.2f}")
    print(f"Type Token Ratio (TTR)   : {stats['type_token_ratio']:.4f}")
    print("======================================")

    if sample_rows:
        print("\nSample tokenized batch rows:")
        for sentence, toks in sample_rows:
            print(f"Original sentence: {sentence}")
            print(f"Tokenized words  : {toks}")
            print("-" * 60)
    print("\nTop 20 Most Frequent Tokens\n")

    for token, freq in token_counter.most_common(20):
        print(f"{token:<25}{freq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())