# https://github.com/spro/practical-pytorch
from typing import List
import torch
import torch.nn as nn
from torch.autograd import Variable
from .helper import Synthesizer
import pickle
import os

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

class Model:

    def __init__(self, rnn_module : nn.Module, optimizer, criterion, synthesizer : Synthesizer):
        self.decoder = rnn_module
        self.optim = optimizer
        self.crit = criterion
        self.synth = synthesizer
        self.model_save_filename = None # type:str

    def train(self, input_tensor : Variable, target_tensor : Variable):
        hidden = self.decoder.init_hidden()
        self.decoder.zero_grad()
        loss = 0

        for c in range(len(input_tensor)):
            output, hidden = self.decoder(input_tensor[c], hidden)
            loss += self.crit(output, target_tensor[c])

        loss.backward()
        self.optim.step()

        return loss.data[0] / len(input_tensor)

    def generate(self, prime_str : List[str], predict_len=100, temperature=0.8):
        hidden = self.decoder.init_hidden()
        prime_input = self.synth.to_tensor(prime_str)
        predicted = [prime_str]

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = self.decoder(prime_input[p], hidden)

        inp = prime_input[-1]

        for p in range(predict_len):
            output, hidden = self.decoder(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_pho = self.synth.alphabet.inv[top_i]
            predicted += predicted_pho
            inp = self.synth.to_tensor(predicted_pho)

        return predicted

    def save(self, filename):
        basename = os.path.splitext(os.path.basename(filename))[0]
        self.model_save_filename = basename + '.pt'
        torch.save(self.decoder, self.model_save_filename)
        with open(basename + ".pckl", "wb") as picklefile:
            # to prevent the decoder from being pickled "vanilla-style", deleting it from the object's field
            # before pickling, then re-adding it
            decoder = self.decoder
            del self.decoder
            pickle.dump(self, picklefile)
            self.decoder = decoder
        print('Saved as %s' % self.model_save_filename)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as pickledfile:
            instance = pickle.load(pickledfile)
            instance.decoder = torch.load(instance.model_save_filename)
        return instance