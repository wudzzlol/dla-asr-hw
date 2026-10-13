from math import log

import torch
from torch.nn.utils.rnn import pad_sequence

from src.datasets.base_dataset import SPECTROGRAM_EPS

LOG_SPECTROGRAM_EPS = log(SPECTROGRAM_EPS)


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    text_encoded_length = []
    spectrogram_length = []
    audio = []
    spectrogram = []
    text = []
    text_encoded = []
    audio_path = []
    for dict_item in dataset_items:
        audio.append(dict_item["audio"].squeeze(0))
        spectrogram.append(dict_item["spectrogram"].squeeze(0).T)
        text.append(dict_item["text"])
        text_encoded.append(dict_item["text_encoded"].squeeze(0))
        audio_path.append(dict_item["audio_path"])
        spectrogram_length.append(dict_item["spectrogram"].shape[-1])
        text_encoded_length.append(dict_item["text_encoded"].shape[-1])

    audio_out = pad_sequence(audio, batch_first=True)
    spectrogram_out = pad_sequence(
        spectrogram, batch_first=True, padding_value=LOG_SPECTROGRAM_EPS
    ).transpose(1, 2)
    text_encoded_out = pad_sequence(text_encoded, batch_first=True)
    text_encoded_length = torch.tensor(text_encoded_length, dtype=torch.long)
    spectrogram_length = torch.tensor(spectrogram_length, dtype=torch.long)

    return {
        "spectrogram": spectrogram_out,
        "spectrogram_length": spectrogram_length,
        "text_encoded": text_encoded_out,
        "text_encoded_length": text_encoded_length,
        "audio": audio_out,
        "text": text,
        "audio_path": audio_path,
    }
