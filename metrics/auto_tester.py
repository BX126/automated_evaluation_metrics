from .diversity.tester import DiversityTest
from .perplexity.tester import PerplexityTest
from .similarity.tester import SimilarityTest

class AutoTest:
    def __init__(self, metrics, prediction):
        self.metrics = metrics
        self.input = prediction
        self.ref = ""
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
            
        if "similarity" in self.metrics:
            results["similarity"] = {}
            temp = {"pred": self.input, "ref": self.ref}
            similarity_scorer = SimilarityTest(temp)
            results["similarity"] = similarity_scorer.text()
        self.results = results
            
    def getResults(self):
        return self.results