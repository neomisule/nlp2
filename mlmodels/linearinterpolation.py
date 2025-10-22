
from .maxlikelihoodest import MLEModel
import math

class LinearInterpolationModel:
    """
    Linear interpolation model combining unigram, bigram, and trigram probabilities.
    Uses weighted sum of probabilities from different N-gram models.
    """
    def __init__(self, lambda1: float, lambda2: float, lambda3: float):
        """
        Initialize the linear interpolation model with weights for each N-gram order.
        Args:
            lambda1 (float): Weight for unigram probability.
            lambda2 (float): Weight for bigram probability.
            lambda3 (float): Weight for trigram probability.
        """
        total = lambda1 + lambda2 + lambda3
        self.lambda1 = lambda1 / total
        self.lambda2 = lambda2 / total
        self.lambda3 = lambda3 / total
        self.unigram_model = MLEModel(1)
        self.bigram_model = MLEModel(2)
        self.trigram_model = MLEModel(3)

    def train(self, text: str):
        """
        Train all component N-gram models on the same text.
        Args:
            text (str): The training text.
        """
        self.unigram_model.train(text)
        self.bigram_model.train(text)
        self.trigram_model.train(text)

    def get_probability(self, ngram: tuple) -> float:
        """
        Calculate the interpolated probability for a trigram.
        Args:
            ngram (tuple): The trigram tuple.
        Returns:
            float: Interpolated probability.
        """
        if len(ngram) == 3:
            trigram_prob = self.trigram_model.get_probability(ngram)
            bigram_prob = self.bigram_model.get_probability(ngram[1:])
            unigram_prob = self.unigram_model.get_probability((ngram[2],))
            return (
                self.lambda3 * trigram_prob +
                self.lambda2 * bigram_prob +
                self.lambda1 * unigram_prob
            )
        else:
            return 0.0

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity using trigrams for the given text.
        Args:
            text (str): The evaluation text.
        Returns:
            float: Perplexity score.
        """
        tokens = self.unigram_model.preprocess_text(text)
        log_prob_sum = 0.0
        total_trigrams = 0
        for i in range(len(tokens) - 2):
            trigram = tuple(tokens[i:i+3])
            prob = self.get_probability(trigram)
            if prob == 0:
                return float('inf')
            log_prob_sum += math.log(prob)
            total_trigrams += 1
        if total_trigrams == 0:
            return float('inf')
        return math.exp(-log_prob_sum / total_trigrams)