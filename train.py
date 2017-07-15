import argparse

import time
import torch
from torch import nn
from tools.model import RNN, Model
from tools.helper import CorpusManager, time_since

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=50)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)


if __name__ == "__main__":
    args = argparser.parse_args()
    corpus = CorpusManager(args.filename)
    synthesizer = corpus.synth

    decoder = RNN(corpus.alphabet_size, args.hidden_size, corpus.alphabet_size, args.n_layers)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model = Model(decoder, decoder_optimizer, criterion, synthesizer)

    loss_avg = 0
    start = time.time()

    try:
        print("Training for %d epochs..." % args.n_epochs)
        for epoch in range(1, args.n_epochs + 1):
            loss = model.train(*corpus.random_training_couple(args.chunk_len))
            loss_avg += loss

            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100))
                print(model.generate(['l'], 100), '\n')

        print("Saving...")
        model.save(args.filename)

    except KeyboardInterrupt:
        print("Saving before quit...")
        model.save(args.filename)
