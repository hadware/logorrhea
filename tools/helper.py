import logging
import math
import random
import time
from typing import List, Dict

import torch
from bidict import bidict
from progressbar import ProgressBar
from torch.autograd import Variable
from voxpopuli import Voice
from voxpopuli.phonemes import Phoneme, PhonemeList


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def mean_and_stddev(values):
    return torch.mean(torch.FloatTensor(values)), torch.std(torch.FloatTensor(values))


class ConsonantStats:

    def __init__(self):
        self.durations = []

    def get_stats(self):
        """Returns mean and standard deviation for each statistical distribution"""
        return {"duration": mean_and_stddev(self.durations)}


class VowelStats:
    def __init__(self):
        self.durations = []
        self.start_pitches = []
        self.end_pitches = []

    def get_stats(self):
        """Returns mean and standard deviation for each statistical distribution"""
        return {"duration": mean_and_stddev(self.durations),
                "start_pitch": mean_and_stddev(self.start_pitches),
                "end_pitch": mean_and_stddev(self.end_pitches)}


class PhonemesStats:

    def __init__(self, phonemes):
        self.phonemes = phonemes
        # consonnants also include the pause phoneme since it's the same kind of distribution
        self.consonants = {pho: ConsonantStats() for pho in self.phonemes.CONSONANTS | "_"} # type:Dict[str,ConsonantStats]
        self.vowels = {pho: VowelStats() for pho in self.phonemes.VOWELS} # type:Dict[str,VowelStats]
        self.vowels_stats = None
        self.consonants_stats = None

    def update(self, phoneme : Phoneme):
        if phoneme.name in self.phonemes.VOWELS:
            pho_stats = self.vowels[phoneme.name]
            pho_stats.durations.append(phoneme.duration)
            pho_stats.start_pitches.append(phoneme.pitch_modifiers[0][1]) # first
            pho_stats.end_pitches.append(phoneme.pitch_modifiers[-1][1]) # last
        else: # it's a consonant
            pho_stats = self.consonants[phoneme.name]
            pho_stats.durations.append(phoneme.duration)

    def aggregate_stats(self):
        self.consonants_stats = {pho: pho_stats.get_stats() for pho, pho_stats in self.consonants.items()}
        self.vowels_stats = {pho: pho_stats.get_stats() for pho, pho_stats in self.vowels.items()}
        

class Synthesizer:

    def __init__(self, phonemes_stats : PhonemesStats, alphabet, voice : Voice):
        self.stats = phonemes_stats
        self.alphabet = alphabet
        self.voice = voice

    def to_tensor(self, phonemes_list : List[str]):
        tensor = torch.zeros(len(phonemes_list)).long()
        for i, pho in enumerate(phonemes_list):
            tensor[i] = self.alphabet[pho]
        return Variable(tensor)

    def gen_vowel_pitches(self, pho : str):
        start_pitch = torch.normal(*self.stats.vowels_stats[pho]["start_pitch"])[0]
        end_pitch = torch.normal(*self.stats.vowels_stats[pho]["end_pitch"])[0]
        return [(0, start_pitch), (80, end_pitch), (100, end_pitch)]

    def synthesize(self, phonemes_list : List[str]) -> bytes:
        """converts a list of phonemes (only their names) to a wav file"""
        pho_list = PhonemeList([])
        for pho in phonemes_list:
            if pho in self.voice.phonems.VOWELS:
                new_phoneme = Phoneme(name=pho,
                                      duration=int(torch.normal(*self.stats.consonants_stats[pho]["duration"])[0]),
                                      pitch_mods=self.gen_vowel_pitches(pho))
            else: #it's a consonant
                new_phoneme = Phoneme(name=pho,
                                      duration=int(torch.normal(*self.stats.consonants_stats[pho]["duration"])[0]),
                                      pitch_mods=[])
            pho_list.append(new_phoneme)
        return self.voice.to_audio(pho_list)


class CorpusManager:

    def __init__(self, filepath: str, lang="fr"):
        self.voice = Voice(lang=lang)
        self.alphabet = bidict({pho : i for i, pho in enumerate(self.voice.phonems | "_")})
        self.alphabet_size = len(self.alphabet)
        phonemized_file = [] # type: List[Phoneme]
        with open(filepath) as corpus_file:
            for line in corpus_file:
                phonemized_file += self.voice.to_phonemes(line.strip())
        self.synth = Synthesizer(self._build_phonemes_stats(phonemized_file), self.alphabet, self.voice)

        self.phonemes_list = [phoneme.name for phoneme in phonemized_file]

    def _build_phonemes_stats(self, phonemized_file):
        logging.info("Building phonemes stats for future synthesizing")
        stats = PhonemesStats(self.voice.phonems)
        bar = ProgressBar()
        for phoneme in bar(phonemized_file):
            stats.update(phoneme)
        stats.aggregate_stats()
        return stats

    def random_training_couple(self, chunk_size):
        start_index = random.randint(0, len(self.phonemes_list) - chunk_size)
        end_index = start_index + chunk_size + 1
        chunk = self.phonemes_list[start_index:end_index]
        inp = self.synth.to_tensor(chunk[:-1])
        target = self.synth.to_tensor(chunk[1:])
        return inp, target

