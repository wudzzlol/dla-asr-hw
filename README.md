# ASR with DeepSpeech2 + CTC (HW1, DLA)

DeepSpeech2-like model (2 Conv2d + 4 BiGRU, 128 log-mel) trained with CTC on LibriSpeech (train-clean-100/360 + train-other-500).
Decoding: argmax, own CTC prefix beam search, and beam search with LibriSpeech 3-gram LM (shallow fusion via `pyctcdecode`)

## Results (char model, beam 50 + 3-gram LM, $\alpha=0.8$, $\beta=3.0$)

| Split | Argmax WER | Beam WER | Beam + LM WER |
|---|---|---|---|
| test-clean | 17.02 | 16.44 | **9.50** |
| test-other | 36.74 | 36.03 | **24.26** |

$\alpha$ and $\beta$ were tuned on dev-other only. Full report with training curves and ablations: [REPORT.md](REPORT.md)

## Installation

```bash
pip install -r requirements.txt
# pyctcdecode pins numpy<2, install without deps
pip install --no-deps pyctcdecode pygtrie
# kenlm: regenerate cython wrapper (needed on recent Python)
pip install "cython>=3"
git clone https://github.com/kpu/kenlm /tmp/kenlm
cd /tmp/kenlm && cython --cplus -3 python/kenlm.pyx -o python/kenlm.cpp && pip install . && cd -
```

Download the final model weights and the language model:

```bash
python scripts/download_weights.py   # -> saved/ds2_full/model_best.pth
python scripts/download_lm.py        # -> data/lm/lm.arpa, data/lm/librispeech-vocab.txt
```

## Demo

`demo.ipynb` runs the full pipeline in Google Colab on any dataset given by a Google Drive link (folder or .zip with `audio/` and optional `transcriptions/`). With an empty link it runs on 9 bundled LibriSpeech utterances from `data/test_custom_dir`

## Inference

On a custom directory:

```bash
python inference.py datasets=custom_dir \
    datasets.test.audio_dir=<path>/audio \
    datasets.test.transcription_dir=<path>/transcriptions \
    metrics=lm \
    text_encoder.lm_model_path=data/lm/lm.arpa \
    text_encoder.vocab_path=data/lm/librispeech-vocab.txt \
    text_encoder.alpha=0.8 text_encoder.beta=3.0 \
    inferencer.decode_method=lm inferencer.beam_size=50 \
    inferencer.save_path=<save_dir>
```

`inferencer.decode_method` can be `argmax`, `beam` or `lm`.
Predictions are saved as `<utterance_id>.txt`.

Metrics between two folders of transcriptions:

```bash
python calc_metrics.py --gt_dir <path>/transcriptions --pred_dir <predictions_dir>
```

## Training

```bash
python train.py -cn=ds2_train          # char model (final)
python -m scripts.train_bpe            # train sentencepiece BPE (vocab 256)
python train.py -cn=ds2_train_bpe      # BPE model (ablation)
```
