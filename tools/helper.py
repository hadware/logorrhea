from voxpopuli import PhonemeList, Voice
from voxpopuli.phonemes import Phoneme
import torch
import logging
from progressbar import ProgressBar
import random
from typing import List


class CorpusManager:

    def __init__(self, filepath : str, lang="fr"):
        self.voice = Voice(lang=lang)
        self.alphabet_size = None
        phonemized_file = [] # type: List[Phoneme]
        with open(filepath) as corpus_file:
            for line in corpus_file:
                phonemized_file += self.voice.to_phonemes(line)
        self._build_phonemes_stats(phonemized_file)

        self.phonemes_list = [phoneme.name for phoneme in phonemized_file]

    def _build_phonemes_stats(self, phonemized_file):
        logging.info("Building phonemes stats for future synthesizing")
        bar = ProgressBar()
        for phoneme in bar(phonemized_file):
            pass

    def _to_tensor(self, phoneme_list : List[str]):
        pass

    def random_training_set(self, chunk_size):
        start_index = random.randint(0, len(self.phonemes_list) - chunk_size)
        end_index = start_index + chunk_len + 1
        chunk = self.phonemes_list[start_index:end_index]
        inp = self._to_tensor(chunk[:-1])
        target = self._to_tensor(chunk[1:])
        return inp, target

    def synthesize(self, phonemes_list : List[str]):
        """converts a list of phonemes (only their names) to a wav file"""
        pass
