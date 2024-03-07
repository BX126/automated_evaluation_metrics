from metrics.auto_tester import AutoTest

"""
supported metrics:
- Diversity
  - Approach: Calculating the number of unique n-grams in the input.
  - Methods:
    - unigram
    - bigram
    - corpus
- Perplexity
  - Approach: Calculating the exponent of mean of log likelihood of all the words in input.
  - Models:
    - gpt2
    - llama2-7b
    - llama2-13b

sample metrics configuration of using all the supported metrics:
{
  "diversity": ["unigram", "bigram", "corpus"],
  "perplexity": ["gpt2", "llama2-7b", "llama2-13b"]
}

"""
 
if __name__ == "__main__":
    input = "Generated Lyrics."
    metrics = {
      "diversity": ["unigram", "bigram", "corpus"],
      "perplexity": ["gpt2", "llama2-7b"]
    }
    auto_test = AutoTest(metrics, input)
    auto_test.test()
    print(auto_test.results)      
            
            
            
