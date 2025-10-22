
from .maxlikelihoodest import MLEModel
import math

class StupidBackoffModel: #trigram -> bigram -> unigram, scaling probabilities by alpha when backing off.
    def __init__(self, alpha: float = 0.4):
        """
        Initialize the Stupid Backoff model.
        Args:
            alpha (float): Scaling factor for backoff (default 0.4).
        """
        self.alpha = alpha
        self.trigram_model = MLEModel(3)
        self.bigram_model = MLEModel(2)
        self.unigram_model = MLEModel(1)

    @property
    def vocab(self):
        # Use the trigram model's vocabulary as the overall vocabulary
        return self.trigram_model.vocab

    def train(self, text: str):
        """
        Train all component N-gram models on the same text.
        Args:
            text (str): The training text.
        """
        self.trigram_model.train(text)
        self.bigram_model.train(text)
        self.unigram_model.train(text)

    def get_probability(self, ngram: tuple, context: tuple = None) -> float:
        """
        Calculate the probability of a trigram using Stupid Backoff.
        Args:
            ngram (tuple): The trigram tuple.
            context (tuple, optional): Ignored for compatibility.
        Returns:
            float: Probability using backoff strategy.
        """
        if len(ngram) == 3:
            # Try trigram first
            trigram_prob = self.trigram_model.get_probability(ngram)
            if trigram_prob > 0:
                return trigram_prob
            # Backoff to bigram
            bigram_prob = self.bigram_model.get_probability(ngram[1:])
            if bigram_prob > 0:
                return self.alpha * bigram_prob
            # Backoff to unigram
            unigram_prob = self.unigram_model.get_probability((ngram[2],))
            return self.alpha * self.alpha * unigram_prob
        return 0.0

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity using trigrams for the given text.
        Args:
            text (str): The evaluation text.
        Returns:
            float: Perplexity score.
        """
        tokens = self.trigram_model.preprocess_text(text)
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