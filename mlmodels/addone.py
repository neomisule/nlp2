
from .ngrambase import NGramLanguageModel

class AddOneSmoothedModel(NGramLanguageModel): #Laplace smoothing - adds one to each N-gram count to avoid zero probabilities for unseen N-grams.
    def get_probability(self, ngram: tuple, context: tuple = None) -> float:
        """
        Calculate the Add-1 smoothed probability of an N-gram.
        Args:
            ngram (tuple): The N-gram tuple whose probability is to be computed.
            context (tuple, optional): The context tuple (ignored, always computed from ngram).
        Returns:
            float: The smoothed probability of the N-gram.
        """
        # For N-gram, context is all but the last token
        context = ngram[:-1] if self.n > 1 else ()
        # Add-one smoothing: add 1 to the count of the N-gram, and add |V| to the context count
        numerator = self.ngram_counts[ngram] + 1
        denominator = self.context_counts[context] + len(self.vocab)
        return numerator / denominator