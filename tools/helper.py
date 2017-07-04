import voxpopuli
import torch

class CorpusManager:

    def __init__(self, filepath : str, lang="fr"):
        self.alphabet_size = None

    def get_random_chunk(self, chunk_size):
        pass

    def synthesize(self, phonemes_list : List[str]):
        """converts a list of phonemes (only their names) to a wav file"""
        pass
