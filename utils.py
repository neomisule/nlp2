import os
from typing import Tuple
from mlmodels.linearinterpolation import LinearInterpolationModel
from mlmodels.stupidbackoff import StupidBackoffModel

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
        
def find_optimal_lambdas(train_text: str, dev_text: str) -> Tuple[float, float, float]:
    """
    Find the optimal lambda values for linear interpolation using development data.
    Args:
        train_text (str): Training text.
        dev_text (str): Development/validation text.
    Returns:
        Tuple[float, float, float]: The best (lambda1, lambda2, lambda3) values.
    """
    best_perplexity = float('inf')
    best_lambdas = (0.33, 0.33, 0.34)
    lambda_combinations = [
        (0.1, 0.1, 0.8), (0.1, 0.2, 0.7), (0.1, 0.3, 0.6),
        (0.2, 0.2, 0.6), (0.2, 0.3, 0.5), (0.2, 0.4, 0.4),
        (0.3, 0.3, 0.4), (0.3, 0.4, 0.3), (0.4, 0.4, 0.2),
        (0.33, 0.33, 0.34), (0.25, 0.25, 0.5), (0.5, 0.25, 0.25),
        (0.15, 0.25, 0.6), (0.25, 0.15, 0.6), (0.35, 0.25, 0.4),
        (0.2, 0.5, 0.3), (0.4, 0.3, 0.3), (0.45, 0.25, 0.3),
        (0.3, 0.5, 0.2), (0.5, 0.3, 0.2), (0.6, 0.2, 0.2)
    ]
    for lambda1, lambda2, lambda3 in lambda_combinations:
        model = LinearInterpolationModel(lambda1, lambda2, lambda3)
        model.train(train_text)
        perplexity = model.calculate_perplexity(dev_text)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_lambdas = (lambda1, lambda2, lambda3)
    return best_lambdas

def find_optimal_alpha(train_text: str, dev_text: str) -> float:
    """
    Find the optimal alpha value for Stupid Backoff using development data.
    Args:
        train_text (str): Training text.
        dev_text (str): Development/validation text.
    Returns:
        float: The best alpha value.
    """
    best_perplexity = float('inf')
    best_alpha = 0.4
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for alpha in alpha_values:
        model = StupidBackoffModel(alpha)
        model.train(train_text)
        perplexity = model.calculate_perplexity(dev_text)
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_alpha = alpha
    return best_alpha

def load_penn_treebank_data():
    """
    Load Penn Treebank data from dataset files.
    Returns:
        Tuple[str, str, str]: (train_text, dev_text, test_text)
    """
    train_text = read_file('data/ptb.train.txt')
    dev_text = read_file('data/ptb.valid.txt')
    test_text = read_file('data/ptb.test.txt')
    return train_text, dev_text, test_text