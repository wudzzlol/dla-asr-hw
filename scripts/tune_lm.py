import argparse
import multiprocessing
from itertools import product

import torch
from hydra.utils import instantiate
from pyctcdecode import build_ctcdecoder
from torch.utils.data import DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from src.datasets import LibrispeechDataset
from src.datasets.collate import collate_fn
from src.metrics.utils import calc_cer, calc_wer


def parse_floats(s):
    return [float(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--part", default="dev-other")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--lm_path", default="data/lm/lm.arpa")
    parser.add_argument("--vocab_path", default="data/lm/librispeech-vocab.txt")
    parser.add_argument("--alphas", type=parse_floats, default="0.3,0.5,0.7,1.0")
    parser.add_argument("--betas", type=parse_floats, default="0.0,0.5,1.0,1.5,2.0")
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    text_encoder = instantiate(ckpt["config"].text_encoder)
    model = instantiate(ckpt["config"].model, n_tokens=len(text_encoder))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    dataset = LibrispeechDataset(
        args.part,
        text_encoder=text_encoder,
        limit=args.limit,
        shuffle_index=True,
        instance_transforms={"get_spectrogram": MelSpectrogram(sample_rate=16000)},
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=2
    )

    log_probs_list, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="acoustic model"):
            out = model(
                spectrogram=batch["spectrogram"].to(device),
                spectrogram_length=batch["spectrogram_length"],
            )
            lp = out["log_probs"].cpu()
            for i, length in enumerate(out["log_probs_length"].cpu()):
                log_probs_list.append(lp[i, : int(length)].numpy())
                targets.append(text_encoder.normalize_text(batch["text"][i]))

    words = open(args.vocab_path, encoding="utf-8").read().split()
    decoder = build_ctcdecoder(
        text_encoder.vocab, kenlm_model_path=args.lm_path, unigrams=words
    )

    results = []
    for alpha, beta in product(args.alphas, args.betas):
        decoder.reset_params(alpha=alpha, beta=beta)
        with multiprocessing.get_context("fork").Pool() as pool:
            preds = decoder.decode_batch(
                pool, log_probs_list, beam_width=args.beam_size
            )
        cer = sum(calc_cer(t, p) for t, p in zip(targets, preds)) / len(targets)
        wer = sum(calc_wer(t, p) for t, p in zip(targets, preds)) / len(targets)
        results.append((wer, cer, alpha, beta))
        print(f"alpha={alpha} beta={beta} WER={wer * 100} CER={cer * 100}")

    results.sort()
    print("Best (by WER):")
    for wer, cer, alpha, beta in results[:5]:
        print(f"alpha={alpha} beta={beta} WER={wer * 100} CER={cer * 100}")


if __name__ == "__main__":
    main()
