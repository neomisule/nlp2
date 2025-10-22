1. Pre-processing and Vocabulary Decisions

Tokenization was performed by splitting text on whitespace and punctuation, converting all text to lowercase. Sentence boundaries were marked using <s> and </s> tokens. Words not present in the training vocabulary were replaced with <unk> for consistent input across all models, reduces vocabulary size, and effective handling of out-of-vocabulary terms.

2. Results
2.1 Maximum Likelihood Estimation N-gram Models

Perplexity for Maximum Likelihood Estimation (unsmoothed) models:

| N-gram           | Perplexity |
| ---------------- | ---------- |
| Unigram (1-gram) | 691.47     |
| Bigram (2-gram)  | ∞          |
| Trigram (3-gram) | ∞          |
| 4-gram           | ∞          |


Observation: Perplexity becomes infinite for higher-order n-grams due to unseen sequences in the test set, highlighting the need for smoothing or backoff strategies.

2.2 Add-1 Smoothed Trigram Model
| Model                     | Perplexity |
| ------------------------- | ---------- |
| Trigram + Add-1 Smoothing | 2930.28    |


Observation: While Add-1 smoothing prevents infinite perplexity, it over-penalizes frequent n-grams, resulting in very high perplexity.

2.3 Linear Interpolation Model
| Model                | Optimal λ (Unigram, Bigram, Trigram) | Perplexity |
| -------------------- | ------------------------------------ | ---------- |
| Linear Interpolation | (0.3, 0.5, 0.2)                      | 212.14     |


Observation: Linear interpolation effectively balances contributions from unigram, bigram, and trigram probabilities, significantly reducing perplexity.

2.4 Stupid Backoff Model
| Model          | Optimal α | Perplexity |
| -------------- | --------- | ---------- |
| Stupid Backoff | 0.9       | 92.11      |


Observation: Stupid Backoff achieves the lowest perplexity by effectively using lower-order statistics when higher-order counts are sparse, without requiring normalization.

2.5 Generated Text (Stupid Backoff)

The following sentences were generated using the best-performing Stupid Backoff model:

1. henry full of <unk> & national security pacific corp inc corp

2. gloomy reports on speculation these positions complained that is also <unk> the proliferation <unk> mania of <unk> from the by mistake he

3. luzon of us into of losses that if law gerald f

4. hatch following the group $ n current copyright law he seeks fines of $ n million in congress but here say the

5. chiefs board stocks after the local soviet newspaper trading activity <unk> considerably across the bay said terms he are starting to be

Observation: The generated text captures some local coherence and realistic phrases, but often includes <unk> tokens and ungrammatical sequences due to vocabulary limitations and the limited context window of n-gram models. Backoff allows the model to generate plausible sequences even when higher-order n-grams are missing.

3. Analysis and Discussion
3.1 Impact of N-gram Order

The MLE results show a clear trend. As n-gram order increases, unsmoothed models encounter infinite perplexity because higher-order n-grams are rarely observed in the training set. This demonstrates data sparsity: the number of possible sequences grows exponentially with N, and limited training data cannot cover all combinations.

The Markov Assumption, which restricts context to N-1 previous words, limits the model’s ability to capture long-range dependencies.

3.2 Comparison of Smoothing and Backoff Strategies
Add-1 Smoothed Trigram - 2930.28	Prevents infinite perplexity but over-penalizes frequent n-grams
Linear Interpolation (λ = 0.3, 0.5, 0.2) - 212.14	Balances contributions from unigram, bigram, and trigram
Stupid Backoff (α = 0.9) - 92.11	Most effective - backs off to lower-order n-grams when higher-order counts are sparse

Discussion:

Unsmooth models have infinite perplexity because they assign zero probability to unseen n-grams.

Add-1 smoothing solves the zero-probability problem but over-penalizes frequent events.

Linear interpolation dramatically reduces perplexity by combining multiple n-gram orders with tuned weights.

Stupid Backoff achieves the lowest perplexity by effectively using lower-order statistics when higher-order counts are sparse, without requiring normalization.

3.3 Qualitative Analysis (Generated Text)

The generated sentences exhibit local coherence (e.g., “current copyright law,” “trading activity”) but lack global fluency.

Frequent <unk> tokens and ungrammatical sequences reflect the limited vocabulary and context of n-gram models.

Backoff allows the model to produce plausible sequences even when higher-order n-grams are missing, explaining why Stupid Backoff produces the most fluent results.
