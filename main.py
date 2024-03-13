import json
import tqdm
import re
from metrics.auto_tester import AutoTest

from inference_util import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


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
- Similarity
  - Approach: Bert score

sample metrics configuration of using all the supported metrics:
{
  "diversity": ["unigram", "bigram", "corpus"],
  "perplexity": ["gpt2", "llama2-7b", "llama2-13b"],
  "similarity": []
}

"""
def matching_rate(prompt, generated_lyrics):
  constriaint = get_music_constraint(prompt)
  return stress_match(generated_lyrics, constriaint)

def get_generated_lyrics(output):
    lyrics_match_single_quote = re.search(
        "generated lyric is '(.*?)<unk>", output)
    return lyrics_match_single_quote.group(1) if lyrics_match_single_quote else None


if __name__ == "__main__":
    metrics = {
        "diversity": ["unigram", "bigram"],
        "perplexity": ["gpt2"]
    }
    auto_test = AutoTest(metrics, "")
    with open("/home/bingxuan/automated_evaluation/generated_fine_tuned_eval_result_without_mapping.json") as json_file:
        results = json.load(json_file)
    response = []
    for result in tqdm.tqdm(results):
        prompt = result['prompt']
        try:
          selected = result['selected']
        except:
          selected = None
        if selected:
            lyrics = selected["sentence"]
            if type(lyrics) == dict:
                lyrics = lyrics["sentence"]
        else:
            continue
        bertscore = selected['bertscore']
        auto_test.setInput(lyrics)
        auto_test.test()
        scores = auto_test.results
        (num_stress,incorrect) = matching_rate(prompt, lyrics)
        scores["similarity"] = bertscore
        scores["matching_rate"] = {"num_stress": num_stress, "incorrect": incorrect}
        response.append({"lyrics": lyrics, "scores": scores})

    with open("/home/bingxuan/automated_evaluation/scores_3.json", "w") as json_file:
        json.dump(response, json_file)
