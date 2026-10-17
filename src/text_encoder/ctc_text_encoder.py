import re
from collections import defaultdict
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

    def beam_search(self, probs, beam_size):
        bs_probs = {(): [1, 0]}  # symb: [blank, non blank]
        probs = probs.tolist()
        for i in range(len(probs)):
            tmp_hyps = defaultdict(lambda: [0.0, 0.0])
            need_to_check = []
            for tok_num in range(len(probs[i])):
                if probs[i][tok_num] >= 1e-3:
                    need_to_check.append(tok_num)
            for tok_num in need_to_check:
                for key in bs_probs.keys():
                    p_b_cur, p_nb_cur = bs_probs[key]
                    if self.ind2char[tok_num] == self.EMPTY_TOK:
                        tmp_hyps[key][0] += (p_b_cur + p_nb_cur) * probs[i][tok_num]
                    elif key and key[-1] == tok_num:
                        tmp_hyps[key][1] += p_nb_cur * probs[i][tok_num]
                        tmp_hyps[key + (tok_num,)][1] += p_b_cur * probs[i][tok_num]
                    else:
                        tmp_hyps[key + (tok_num,)][1] += (p_b_cur + p_nb_cur) * probs[
                            i
                        ][tok_num]
            tmp_hyps = dict(
                sorted(tmp_hyps.items(), key=lambda x: -sum(x[1]))[:beam_size]
            )
            sm = 0.0
            for hyp, hyp_probs in tmp_hyps.items():
                sm += hyp_probs[0] + hyp_probs[1]
            for hyp in tmp_hyps.keys():
                tmp_hyps[hyp][0] /= sm
                tmp_hyps[hyp][1] /= sm
            bs_probs = tmp_hyps
        best_hyp = ()
        best_sm = -1
        for hyp, hyp_probs in bs_probs.items():
            if sum(hyp_probs) > best_sm:
                best_sm = sum(hyp_probs)
                best_hyp = hyp
        return (
            "".join([self.ind2char[i] for i in best_hyp])
            .replace(self.BPE_SPACE, " ")
            .strip()
        )

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
