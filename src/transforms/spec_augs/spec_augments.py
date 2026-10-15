import torchaudio
from torch import Tensor, nn

from src.datasets.collate import LOG_SPECTROGRAM_EPS


class SpectrogramAugmentations(nn.Module):
    def __init__(self, freq_mask_param, time_mask_param):
        super().__init__()
        self._time_aug = torchaudio.transforms.TimeMasking(
            time_mask_param, iid_masks=True
        )
        self._freq_aug = torchaudio.transforms.FrequencyMasking(
            freq_mask_param, iid_masks=True
        )

    def forward(self, spectrogram: Tensor):
        spectrogram = spectrogram.unsqueeze(1)
        spectrogram = self._freq_aug(spectrogram, LOG_SPECTROGRAM_EPS)
        spectrogram = self._time_aug(spectrogram, LOG_SPECTROGRAM_EPS)
        spectrogram = spectrogram.squeeze(1)
        return spectrogram
