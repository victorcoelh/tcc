import nltk
import numpy as np
from bert_score import BERTScorer
from nltk.translate import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

spice_scorer = Spice()
bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)


def calculate_cider(references: list[list[str]], candidates: list[str],
                    batch_size: int) -> list[float]:
    reference: dict[int, list[str]] = {}
    predicted: dict[int, list[str]] = {}
    
    for i in range(batch_size):
        predicted[i] = [" ".join(nltk.word_tokenize(candidates[i]))]
        reference[i] = [" ".join(nltk.word_tokenize(reference))
                        for reference in references[i]]
        
    return list(Cider(n=4, sigma=6.0).compute_score(reference, predicted)[1]) # type: ignore


def calculate_meteor(references: list[list[str]], candidates: list[str],
                     batch_size: int) -> list[float]:
    scores = []
    for i in range(batch_size):
        reference_tokens = [nltk.word_tokenize(ref) for ref in references[i]]
        generated_tokens = nltk.word_tokenize(candidates[i])
        scores.append(meteor_score.meteor_score(reference_tokens, generated_tokens))
        
    return scores


def calculate_spice(references: list[list[str]], candidates: list[str],
                    batch_size: int) -> list[float]:
    reference: dict[int, list[str]] = {}
    predicted: dict[int, list[str]] = {}
    
    for i in range(batch_size):
        predicted[i] = [candidates[i]]
        reference[i] = references[i]

    _, scores = spice_scorer.compute_score(reference, predicted)
    return [score["All"]["f"] for score in scores]


def calculate_bertscore(references: list[list[str]], candidates: list[str],
                        _batch_size: int) -> list[float]:
    _, _, f1 = bert_scorer.score(candidates, references)
    return f1.tolist() # type: ignore
