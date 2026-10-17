import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pred_dir", required=True, help="Folder with predicted transcriptions"
)
parser.add_argument(
    "--gt_dir", required=True, help="Folder with ground truth transcriptions"
)
args = parser.parse_args()

cers, wers = [], []
for gt_path in sorted(Path(args.gt_dir).glob("*.txt")):
    pred_path = Path(args.pred_dir) / gt_path.name
    if not pred_path.exists():
        print(f"Predicted path - {pred_path} - does not exists")
        continue
    gt_text = CTCTextEncoder.normalize_text(gt_path.read_text(encoding="utf-8").strip())
    pred_text = pred_path.read_text(encoding="utf-8").strip()
    cers.append(calc_cer(gt_text, pred_text))
    wers.append(calc_wer(gt_text, pred_text))
print(f"Quantity of files: {len(cers)}")
try:
    print(f"CER: {sum(cers) / len(cers)}")
    print(f"WER: {sum(wers) / len(wers)}")
except ZeroDivisionError:
    print("No matching files found between gt_dir and pred_dir")
