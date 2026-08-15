from datasets import load_dataset
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', '-o', default='data\\gujarati.txt')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--flush_every', type=int, default=1000)
args = parser.parse_args()

out_path = args.output
os.makedirs(os.path.dirname(out_path), exist_ok=True)

print('Streaming IndicCorpV2 gujarati and writing to', out_path)

ds = load_dataset('ai4bharat/IndicCorpV2', 'indiccorp_v2', split='guj_Gujr', streaming=True)
count = 0
with open(out_path, 'w', encoding='utf-8') as f:
    for ex in ds:
        text_val = None
        if isinstance(ex, dict):
            candidates = ['text','sentence','content','translation','line']
            for k in candidates:
                if k in ex and isinstance(ex[k], str):
                    text_val = ex[k]
                    break
            if text_val is None:
                for k,v in ex.items():
                    if isinstance(v, str):
                        text_val = v
                        break
                    if isinstance(v, (list,tuple)) and v and all(isinstance(i,str) for i in v):
                        text_val = ' '.join(v)
                        break
        else:
            text_val = str(ex)
        if not text_val:
            continue
        f.write(text_val.replace('\n',' ') + '\n')
        count += 1
        if args.limit and count >= args.limit:
            break
        if count % args.flush_every == 0:
            f.flush()
            print(f'Wrote {count} lines...', flush=True)

print('Done. Total lines written:', count)
