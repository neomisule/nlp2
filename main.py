
from mlmodels.maxlikelihoodest import MLEModel
from mlmodels.addone import AddOneSmoothedModel
from mlmodels.linearinterpolation import LinearInterpolationModel
from mlmodels.stupidbackoff import StupidBackoffModel
from gen import TextGenerator
from utils import find_optimal_lambdas, find_optimal_alpha, load_penn_treebank_data

def main():
    """
    To train and evaluate various N-gram language models and generate text.
    It will train and evaluate MLE models (unigram, bigram, trigram, 4-gram),
    Add-1 smoothed trigram model, linear interpolation model (with optimal lambdas),
    Stupid Backoff model (with optimal alpha), generate sentences using the best model.
    """
    train_text, dev_text, test_text = load_penn_treebank_data()

    # Maximum likelihood estimation Models (N=1,2,3,4)
    print("\n1. Training MLE Models...")
    for n in [1, 2, 3, 4]:
        model = MLEModel(n)
        model.train(train_text)
        perplexity = model.calculate_perplexity(test_text) #some model will have 'INF' perplexity cause of handling zeroes error due to unseen grams in the test dataset
        print(f"MLE {n}-gram perplexity: {perplexity}")

    # Add-1 Smoothing
    print("\n2. Training Add-1 Smoothed Trigram Model...")
    add_one_model = AddOneSmoothedModel(3)
    add_one_model.train(train_text)
    add_one_perplexity = add_one_model.calculate_perplexity(test_text)
    print(f"Add-1 smoothed trigram perplexity: {add_one_perplexity}")

    # Linear Interpolation
    print("\n3. Training Linear Interpolation Model...")
    optimal_lambdas = find_optimal_lambdas(train_text, dev_text)
    print(f"Optimal lambdas: {optimal_lambdas}")
    interp_model = LinearInterpolationModel(*optimal_lambdas)
    interp_model.train(train_text)
    interp_perplexity = interp_model.calculate_perplexity(test_text)
    print(f"Linear interpolation perplexity: {interp_perplexity}")

    # Stupid Backoff
    print("\n4. Training Stupid Backoff Model...")
    optimal_alpha = find_optimal_alpha(train_text, dev_text)
    print(f"Optimal alpha: {optimal_alpha}")
    backoff_model = StupidBackoffModel(optimal_alpha)
    backoff_model.train(train_text)
    backoff_perplexity = backoff_model.calculate_perplexity(test_text)
    print(f"Stupid backoff perplexity: {backoff_perplexity}")

    # Text Generation
    print("\n5. Generating Text...")
    generator = TextGenerator(backoff_model)
    print("\nGenerated Sentences:")
    for i in range(5):
        sentence = generator.generate_text(max_length=22
                                           )
        print(f"{i+1}. {sentence}")

if __name__ == "__main__":
    main()