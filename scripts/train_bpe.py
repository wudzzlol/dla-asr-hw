import json

import sentencepiece as spm

from src.text_encoder import CTCTextEncoder

texts = []
paths = [
    "data/bpe_texts/train-clean-100_index.json",
    "data/bpe_texts/train-clean-360_index.json",
    "data/bpe_texts/train-other-500_index.json",
]
for path in paths:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            texts.append(CTCTextEncoder.normalize_text(item["text"]))

with open("data/bpe_texts/all_texts.txt", "w", encoding="utf-8") as f:
    f.writelines(text + "\n" for text in texts)

spm.SentencePieceTrainer.train(
    input="data/bpe_texts/all_texts.txt",
    model_prefix="src/text_encoder/bpe",
    vocab_size=256,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=-1,
    eos_id=-1,
)
