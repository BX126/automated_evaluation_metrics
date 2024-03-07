from .diversity.tester import DiversityTest
from .perplexity.tester import PerplexityTest

class AutoTest:
    def __init__(self, metrics, input):
        self.metrics = metrics
        self.input = input
        self.results = {}
    
    def setInput(self, input):
        self.input = input

    def test(self):
        results = {}
        if "diversity" in self.metrics:
          results["diversity"] = {}
          diversity_scorer = DiversityTest(self.input)
          for mode in self.metrics["diversity"]:
            results["diversity"][mode] = diversity_scorer.test(mode)
        
        if "perplexity" in self.metrics:
            results["perplexity"] = {}
            perplexity_scorer = PerplexityTest(self.input, self.metrics["perplexity"])
            results["perplexity"] = perplexity_scorer.test()
        self.results = results
            
    def getResults(self):
        return self.results