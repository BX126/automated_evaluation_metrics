from .utils import *

class DiversityTest:
    def __init__(self, input):
        self.input = input
      
    def setInput(self, input):
        self.input = input
        
    def test(self, mode, n=1):
        if mode == "unigram":
            return distinct_n_sentence_level(self.input, 1)
        elif mode == "bigram":
            return distinct_n_sentence_level(self.input, 2)
        elif mode == "corpus":
            return distinct_n_corpus_level(self.input, n)
        else:
            raise ValueError("Invalid mode.")