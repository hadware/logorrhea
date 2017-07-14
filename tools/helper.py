import logging
import math
import random
import time
from typing import List

import torch
from bidict import bidict
from progressbar import ProgressBar
from torch.autograd import Variable
from voxpopuli import Voice
from voxpopuli.phonemes import Phoneme


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Synthesizer:

    def __init__(self, phonemes_stats : PhonemeStats , alphabet):
        self.stats = phonemes_stats
        self.alphabet = alphabet

    def to_tensor(self, phonemes_list : List[str]):
        tensor = torch.zeros(len(phonemes_list)).long()
        for i, pho in enumerate(phonemes_list):
            tensor[i] = self.alphabet[pho]
        return Variable(tensor)

    def synthesize(self, phonemes_list : List[str]):
        """converts a list of phonemes (only their names) to a wav file"""
        pass


class CorpusManager:

    def __init__(self, filepath: str, lang="fr"):
        self.voice = Voice(lang=lang)
        self.alphabet = bidict({pho : i for i, pho in enumerate(self.voice.phonems | "_")})
        self.alphabet_size = len(self.alphabet)
        phonemized_file = [] # type: List[Phoneme]
        with open(filepath) as corpus_file:
            for line in corpus_file:
                phonemized_file += self.voice.to_phonemes(line.strip())
        self._build_phonemes_stats(phonemized_file)
        self.synth = Synthesizer(self._build_phonemes_stats(phonemized_file), alphabet)

        self.phonemes_list = [phoneme.name for phoneme in phonemized_file]

    def _build_phonemes_stats(self, phonemized_file):
        logging.info("Building phonemes stats for future synthesizing")
        bar = ProgressBar()
        for phoneme in bar(phonemized_file):
            pass


    def random_training_couple(self, chunk_size):
        start_index = random.randint(0, len(self.phonemes_list) - chunk_size)
        end_index = start_index + chunk_size + 1
        chunk = self.phonemes_list[start_index:end_index]
        inp = self.synth.to_tensor(chunk[:-1])
        target = self.synth.to_tensor(chunk[1:])
        return inp, target

