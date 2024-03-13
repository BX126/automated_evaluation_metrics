from evaluate import load

class SimilarityTest:
    def __init__(self, inputs):
      self.bertscore = load("bertscore")
      self.prediction = inputs["pref"]
      self.reference = inputs["ref"]
      
    def setInput(self, inputs):
      self.prediction = inputs["pref"]
      self.reference = inputs["ref"]

    def test(self):
      return self.bertscore.compute(predictions=self.prediction, references=self.reference, lang="en")
      