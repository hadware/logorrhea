import argparse
import voxpopuli
import subprocess
from progressbar import ProgressBar

argparser = argparse.ArgumentParser()
argparser.add_argument('filepath', type=str)
argparser.add_argument('outfile', type=str)
argparser.add_argument('--lang', type=str, help="Languge of the phonemization", default="fr")


def file_linecount(filepath):
    wc_output = subprocess.getoutput("wc -l %s" % filepath)
    return int(wc_output.split()[0])


if __name__ == "__main__":
    args = argparser.parse_args()

    bar = ProgressBar(max_value=file_linecount(args.filepath) + 2)
    voice = voxpopuli.Voice(lang=args.lang)
    with open(args.filepath) as corpus_file, open(args.outfile, "w") as outfile:
        for line in bar(corpus_file):
            outfile.write(str(voice.to_phonemes(line)) + "\n")