
from typing import List
import random

class TextGenerator: #Generates text by sampling from the model's probability distribution.
    def __init__(self, model):
        """
        Initialize the text generator with a language model (must have vocab and get_probability).
        """
        self.model = model

    def generate_text(self, max_length: int = 30, start_tokens: List[str] = None) -> str:
        """
        Generate text using the language model.
        Args:
            max_length (int): Maximum number of tokens to generate.
            start_tokens (List[str], optional): Initial tokens to start generation.
        Returns:
            str: The generated text as a string.
        """
        if start_tokens is None:
            start_tokens = ['<s>', '<s>']  # Default: trigram start
        generated = start_tokens.copy()
        for _ in range(max_length):
            # Use last two tokens as context (for trigram models)
            if len(generated) >= 2:
                context = tuple(generated[-2:])
            else:
                context = tuple(generated)
            next_word_probs = []
            for word in self.model.vocab:
                if word == '<s>':
                    continue  # Skip start token
                if len(generated) >= 2:
                    ngram = context + (word,)
                else:
                    ngram = (word,)
                prob = self.model.get_probability(ngram, context)
                if prob > 0:
                    next_word_probs.append((word, prob))
            if not next_word_probs:
                break  # No valid next words
            words, probs = zip(*next_word_probs)
            next_word = random.choices(words, weights=probs, k=1)[0]
            generated.append(next_word)
            if next_word == '</s>':
                break  # End of sentence
        # Remove special tokens from output
        clean_text = [token for token in generated if token not in ['<s>', '</s>']]
        return ' '.join(clean_text)