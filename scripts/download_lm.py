import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

LM_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
VOCAB_URL = "https://www.openslr.org/resources/11/librispeech-vocab.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="data/lm")
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
gz_path = out_dir / "3-gram.pruned.1e-7.arpa.gz"
lm_path = out_dir / "lm.arpa"
vocab_path = out_dir / "librispeech-vocab.txt"

if not lm_path.exists():
    print("Downloading LM...")
    urllib.request.urlretrieve(LM_URL, gz_path)
    # the original LM is uppercase, our vocabulary is lowercase
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(
        lm_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            fout.write(line.lower())
    gz_path.unlink()

if not vocab_path.exists():
    print("Downloading vocab...")
    urllib.request.urlretrieve(VOCAB_URL, vocab_path)
    text = vocab_path.read_text(encoding="utf-8").lower()
    vocab_path.write_text(text, encoding="utf-8")

print(f"LM: {lm_path}\nVocab: {vocab_path}")
