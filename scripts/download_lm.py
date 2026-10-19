import argparse
import gzip
import urllib.request
from pathlib import Path

LM_URL = "https://us.openslr.org/resources/11/3-gram.arpa.gz"
VOCAB_URL = "https://us.openslr.org/resources/11/librispeech-vocab.txt"


def progress(block, block_size, total):
    if total > 0 and block % 2000 == 0:
        print(f"{block * block_size / 1e6} / {total / 1e6} MB", flush=True)


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="data/lm")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
gz_path = out_dir / "3-gram.arpa.gz"
lm_path = out_dir / "lm.arpa"
vocab_path = out_dir / "librispeech-vocab.txt"

if not lm_path.exists():
    print("Downloading LM...", flush=True)
    urllib.request.urlretrieve(LM_URL, gz_path, reporthook=progress)
    print("Lowercasing LM...", flush=True)
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(
        lm_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            fout.write(line.lower())
    gz_path.unlink()

if not vocab_path.exists():
    print("Downloading vocab...", flush=True)
    urllib.request.urlretrieve(VOCAB_URL, vocab_path)
    text = vocab_path.read_text(encoding="utf-8").lower()
    vocab_path.write_text(text, encoding="utf-8")

print(f"LM: {lm_path}\nVocab: {vocab_path}")
