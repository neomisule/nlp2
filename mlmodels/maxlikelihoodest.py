
from .ngrambase import NGramLanguageModel

class MLEModel(NGramLanguageModel):
    """
    Maximum Likelihood Estimation (MLE) N-gram model.
    Estimates probability of an N-gram as the count of the N-gram divided by the count of its context.
    Inherits from NGramLanguageModel.
    """

    def get_probability(self, ngram: tuple, context: tuple = None) -> float:
        """
        Calculate the MLE probability of an N-gram.
        Args:
            ngram (tuple): The N-gram tuple whose probability is to be computed.
            context (tuple, optional): The context tuple (ignored, always computed from ngram).
        Returns:
            float: The MLE probability of the N-gram.
        """
        context = ngram[:-1] if self.n > 1 else ()
        if self.context_counts[context] == 0:
            return 0.0
        return self.ngram_counts[ngram] / self.context_counts[context]