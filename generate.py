import argparse

import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('-p', '--prime_str', type=str, default='A')
argparser.add_argument('-l', '--predict_len', type=int, default=100)
argparser.add_argument('-t', '--temperature', type=float, default=0.8)

if __name__ == "__main__":
    args = argparser.parse_args()

    decoder = torch.load(args.filename)