import re
from string import ascii_lowercase

import sentencepiece as spm
import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""
    BPE_SPACE = "\u2581"

    def __init__(self, alphabet=None, bpe_path=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        if bpe_path is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_path)
            self.vocab = [self.EMPTY_TOK] + [
                self.sp.id_to_piece(i) for i in range(len(self.sp))
            ]
        else:
            self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
            self.sp = None

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        if self.sp is None:
            try:
                return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
            except KeyError:
                unknown_chars = set(
                    [char for char in text if char not in self.char2ind]
                )
                raise Exception(
                    f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
                )
        else:
            encoded_text = torch.Tensor(self.sp.encode(text))
            encoded_text += 1
            return encoded_text.unsqueeze(0)

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return (
            "".join([self.ind2char[int(ind)] for ind in inds])
            .replace(self.BPE_SPACE, " ")
            .strip()
        )

    def ctc_decode(self, inds) -> str:
        res = []
        cur = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            ind = int(ind)
            if ind != cur:
                if cur != self.char2ind[self.EMPTY_TOK]:
                    res.append(self.ind2char[cur])
                cur = ind
        if cur != self.char2ind[self.EMPTY_TOK]:
            res.append(self.ind2char[cur])
        return "".join(res).replace(self.BPE_SPACE, " ").strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
