import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PerplexityTest:
    def __init__(self, inputs, models):
      self.inputs = inputs
      self.models_zoo = {
        "gpt2": "gpt2",
        "llama2-7b": "NousResearch/Llama-2-7b-hf",
        "llama2-13b": "NousResearch/Llama-2-13b-hf"
      }
      print("Loading models...")
      print("(It may take a while to download the models for the first time)")
      self.model_names = models
      self.models = [AutoModelForCausalLM.from_pretrained(self.models_zoo[model]) for model in models]
      self.tokenizers = [AutoTokenizer.from_pretrained(self.models_zoo[model]) for model in models]
      
    def setInput(self, inputs):
      self.inputs = inputs

    def test(self):
      results = {}
      for i, model in enumerate(self.models):
        tokenizer = self.tokenizers[i]
        input_ids = tokenizer.encode(self.inputs, return_tensors="pt")
        with torch.no_grad():
          output = model(input_ids, labels=input_ids)
        loss = output.loss
        results[self.model_names[i]] = torch.exp(loss).item()
      return results