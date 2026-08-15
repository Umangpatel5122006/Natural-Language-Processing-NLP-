# Assignment 1 — Gujarati Corpus Tokenization and Statistics

## Source data
- Hugging Face: <https://huggingface.co/datasets/ai4bharat/IndicCorpV2>
- Language extracted: **Gujarati** (`gu`).
- The IndicCorpV2 Gujarati dump (`gu.txt`, ~14 GB, 43.6 M paragraphs) is read
  streaming, so RAM stays flat regardless of corpus size.
- Large corpus files are intentionally excluded from GitHub tracking to keep the repository under the 100 MB upload limit. Download or generate them locally before running the full pipeline.

## Pipeline
Implemented in `scripts/gujarati_pipeline.py`.

1. **Load** — read the input UTF-8 text line-by-line (paragraph per line).
2. **Sentence tokenize** — split each paragraph on terminators
   (`.`, `?`, `!`, Gujarati danda `U+0964`, double danda `U+0965`),
   preserving URLs, e-mails, dates, times, and decimals via a
   mask/unmask pass so the sentence splitter doesn't tear them apart.
3. **Word tokenize** — split each sentence on whitespace and
   Gujarati/ASCII punctuation; protected spans (URLs, e-mails, dates,
   times, decimals) are masked before splitting and restored in the
   resulting tokens, so:
   - `https://example.com/page?id=42` stays one token,
   - `test@example.com` stays one token,
   - `૧૨.૫૬` (Gujarati decimal) stays one token,
   - `25/01/2024`, `10:30` stay one token.
4. **Save** — write tokenized sentences as one sentence per line into
   `outputs/tokenized_sentences.txt` and the same rows
   (`sentence_id`, `sentence`, `tokens`, `token_count`) into
   `outputs/tokenized_sentences.parquet` (snappy compression).
5. **Stats** — compute and persist corpus statistics to
   `outputs/corpus_stats.json`.

## How to run

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\myenv\Scripts\python.exe Assignment_1\scripts\gujarati_pipeline.py
```

Environment-variable overrides:

| variable          | default                                                                       |
| ----------------- | ----------------------------------------------------------------------------- |
| `INPUT_PATH`      | `E:\STUDY\NLP\Language\gu.txt`                  |
| `OUTPUT_TXT`      | `Assignment_1\outputs\tokenized_sentences.txt`                                |
| `OUTPUT_DETAILED_TXT` | `Assignment_1\outputs\tokenized_sentences_with_source.txt`               |
| `OUTPUT_PARQUET`  | `Assignment_1\outputs\tokenized_sentences.parquet`                            |
| `OUTPUT_STATS`    | `Assignment_1\outputs\corpus_stats.json`                                     |
| `MAX_LINES`       | `50000` (default sample size). Set to `0` for no limit or `50000` to cap the first N paragraphs. |

The script defaults to processing the first 50,000 paragraphs. To cap at the first 50,000 paragraphs explicitly:

```powershell
$env:MAX_LINES = '50000'
.\myenv\Scripts\python.exe Assignment_1\scripts\gujarati_pipeline.py
```

## Latest run — first 50,000 Gujarati paragraphs

Outputs (see `outputs/`):

| file                          | size       |
| ----------------------------- | ---------- |
| `tokenized_sentences.txt`     | 15.49 MB   |
| `tokenized_sentences.parquet` | 13.59 MB   |
| `corpus_stats.json`           | 565 B      |

Statistics from `outputs/corpus_stats.json`:

| metric                       | value          |
| ---------------------------- | -------------- |
| total sentences              | 70,750         |
| total words                  | 1,053,615      |
| total characters             | 6,185,457      |
| average sentence length      | 14.8921 words  |
| average word length          | 4.8256 chars   |
| type / token ratio (TTR)     | 0.107896       |

## File layout

```
Assignment_1/
├── README.md
├── scripts/
│   └── gujarati_pipeline.py     # main pipeline (stream + tokenize + save + stats)
└── outputs/
    ├── tokenized_sentences.txt  # one tokenized sentence per line (required format)
    ├── tokenized_sentences.parquet  # same rows, columnar + snappy (compression)
    ├── corpus_stats.json        # the six required statistics
    └── run_log.txt              # captured stdout from the last run
```