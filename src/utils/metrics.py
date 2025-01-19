import nltk
from nltk.translate import meteor_score
from pycocoevalcap.cider.cider import Cider


def calculate_cider(references: list[list[str]], candidates: list[str], batch_size: int) -> float:
    reference: dict[int, list[str]] = {}
    predicted: dict[int, list[str]] = {}
    
    for i in range(batch_size):
        predicted[i] = [" ".join(nltk.word_tokenize(candidates[i]))]
        reference[i] = [" ".join(nltk.word_tokenize(reference))
                        for reference in references[i]]
        
    return Cider(n=4, sigma=6.0).compute_score(reference, predicted)[0] # type: ignore
    
def calculate_meteor(references: list[list[str]], candidates: list[str], batch_size: int) -> float:
    score = 0
    for i in range(batch_size):
        reference_tokens = [nltk.word_tokenize(ref) for ref in references[i]]
        generated_tokens = nltk.word_tokenize(candidates[i])
        score += meteor_score.meteor_score(reference_tokens, generated_tokens)
        
    return score / batch_size
