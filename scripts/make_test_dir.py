import argparse
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/datasets/librispeech/dev-clean")
parser.add_argument("--dst", default="data/test_custom_dir")
parser.add_argument("--n", type=int, default=10)
args = parser.parse_args()

src, dst = Path(args.src), Path(args.dst)
(dst / "audio").mkdir(parents=True, exist_ok=True)
(dst / "transcriptions").mkdir(parents=True, exist_ok=True)

count = 0
for trans_file in sorted(src.rglob("*.trans.txt")):
    for line in trans_file.read_text(encoding="utf-8").splitlines():
        utt_id, text = line.split(" ", 1)
        shutil.copy(
            trans_file.parent / f"{utt_id}.flac", dst / "audio" / f"{utt_id}.flac"
        )
        (dst / "transcriptions" / f"{utt_id}.txt").write_text(
            text.lower(), encoding="utf-8"
        )
        count += 1
        if count >= args.n:
            break
    if count >= args.n:
        break

print(f"Created {count} utterances in {dst}")
