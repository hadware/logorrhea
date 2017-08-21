# Logorrhea

Using the latest advance in Deep Learning and two old TTS softwares to generate what should look like speech.

Mostly inspired by [Andrew Karpathy's article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), and [this tutorial](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb) from the PyTorch library.
Basically, instead of doing the generation at character-level, I'm doing it at the phoneme level.
Espeak takes care of phonemizing the input text with some very efficient rules, and mbrola takes the generated phonems to make actual sound. 

Also, uses my own [Voxpopuli](https://github.com/hadware/voxpopuli) library to do the phonemization and sound rendering stuff.