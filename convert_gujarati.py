"""Convert a local Gujarati text file into a Parquet file using repo-relative paths."""
import io
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT = PROJECT_ROOT / 'Language' / 'gu.txt'
OUTPUT = PROJECT_ROOT / 'Temp' / 'gujarati.parquet'
BATCH = 50_000

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

if not INPUT.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT}. Provide a corpus file or override INPUT_PATH in the script."
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
print(f'Input : {INPUT}')
print(f'Output: {OUTPUT}')
print(f'Reading in batches of {BATCH:,} lines ...\n')

schema = pa.schema([('text', pa.string())])
writer = pq.ParquetWriter(str(OUTPUT), schema, compression='snappy')

buf_text = []
total = 0
t0 = time.time()

with INPUT.open('r', encoding='utf-8', errors='replace') as f:
    for line in f:
        s = line.rstrip('\n').rstrip('\r')
        if s:
            buf_text.append(s)
        if len(buf_text) >= BATCH:
            tbl = pa.Table.from_pandas(pd.DataFrame({'text': buf_text}), schema=schema, preserve_index=False)
            writer.write_table(tbl)
            total += len(buf_text)
            buf_text.clear()
            elapsed = time.time() - t0
            print(f'  {total:>12,} lines  ({elapsed:6.1f}s)')

if buf_text:
    tbl = pa.Table.from_pandas(pd.DataFrame({'text': buf_text}), schema=schema, preserve_index=False)
    writer.write_table(tbl)
    total += len(buf_text)
    buf_text.clear()

writer.close()
elapsed = time.time() - t0

size_mb = OUTPUT.stat().st_size / (1024 * 1024)
print(f'\nDone. Wrote {total:,} non-empty lines to {OUTPUT}')
print(f'Parquet size: {size_mb:.2f} MB   (elapsed {elapsed:.1f}s)')

df = pd.read_parquet(OUTPUT)
print(f'Roundtrip read OK: {len(df):,} rows')
print(df.head())
