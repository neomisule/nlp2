
import math
import re
from collections import defaultdict
from typing import List, Set

class NGramLanguageModel:
    """
    Base class for N-gram language models.
    Provides methods for preprocessing text, training N-gram counts, and calculating perplexity.
    Subclasses must implement get_probability().
    """
    def __init__(self, n: int, vocab: Set[str] = None):
        """
        Initialize the N-gram model.
        Args:
            n (int): The order of the N-gram (e.g., 1=unigram, 2=bigram, etc.)
            vocab (set, optional): Initial vocabulary set. If None, will be built from training data.
        """
        self.n = n
        self.vocab = vocab or set()
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by tokenizing, lowercasing, and adding sentence boundaries.
        Args:
            text (str): The input text.
        Returns:
            List[str]: List of tokens with sentence boundaries.
        """
        # Add spaces around sentence-ending punctuation
        text = re.sub(r'([.!?])', r' \1 ', text)
        # Tokenize on whitespace and punctuation
        tokens = re.findall(r'\S+', text.lower())
        processed = []
        in_sentence = False
        for token in tokens:
            if token in ['.', '!', '?']:
                processed.append('</s>')  # End of sentence
                in_sentence = False
            else:
                if not in_sentence:
                    processed.extend(['<s>'] * (self.n - 1))  # Add start tokens
                    in_sentence = True
                processed.append(token)
        # Close final sentence if not already closed
        if in_sentence:
            processed.append('</s>')
        return processed

    def train(self, text: str):
        """
        Train the N-gram model by counting N-grams and contexts in the text.
        Args:
            text (str): The training text.
        """
        tokens = self.preprocess_text(text)
        self.vocab.update(tokens)
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = ngram[:-1] if self.n > 1 else ()
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def get_probability(self, ngram: tuple, context: tuple = None) -> float:
        """
        Get the probability of an N-gram. Must be implemented by subclasses.
        Args:
            ngram (tuple): The N-gram tuple.
            context (tuple, optional): The context tuple.
        Returns:
            float: Probability of the N-gram.
        """
        raise NotImplementedError("Subclasses must implement get_probability().")

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate the perplexity of the model on the given text.
        Args:
            text (str): The evaluation text.
        Returns:
            float: Perplexity score.
        """
        tokens = self.preprocess_text(text)
        log_prob_sum = 0.0
        total_ngrams = 0
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = ngram[:-1] if self.n > 1 else ()
            prob = self.get_probability(ngram, context)
            if prob == 0:
                return float('inf')  # Infinite perplexity if any probability is zero
            log_prob_sum += math.log(prob)
            total_ngrams += 1
        if total_ngrams == 0:
            return float('inf')
        return math.exp(-log_prob_sum / total_ngrams)