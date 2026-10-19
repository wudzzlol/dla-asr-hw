import argparse
from pathlib import Path

import gdown

WEIGHTS = {
    "ds2_full": "1pTRNsmriUzyTYMM7r99k9EDys6tx4o-S",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ds2_full", choices=list(WEIGHTS))
    args = parser.parse_args()

    out = Path("saved") / args.model / "model_best.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists, skipping")
        return
    gdown.download(id=WEIGHTS[args.model], output=str(out), quiet=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
