import sys
import numpy as np
from models import PriorModel # <-- Implemented as a uniform distribution.
from models import TranslationModel # <-- Not implemented 
from models import TransitionModel # <-- You will need this for an HMM.
from utils import read_all_tokens, output_alignments_per_test_set
from nltk.stem import WordNetLemmatizer

def get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model, trasition_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    alignment_posteriors = np.zeros((len(trg_tokens), len(src_tokens)))
    translation_prob = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)
    prior_prob = prior_model.get_parameters_for_sentence_pair(len(src_tokens), len(trg_tokens))

    transition_posteriors = np.zeros((len(src_corpus), len(src_corpus)))

    alignment_posteriors += prior_prob.T * translation_prob.T
    denominator = np.reshape(np.sum(alignment_posteriors, axis=1), (len(trg_tokens), 1))
    alignment_posteriors /= denominator

    answers = np.argmax(alignment_posteriors, axis=1)
    arange = np.arange(len(trg_tokens))
    answers_probs = alignment_posteriors[arange, answers]
    log_likelihood = np.sum(np.log(answers_probs))
    return alignment_posteriors, transition_posteriors, log_likelihood, answers


def collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model, trasition_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # Infer posterior
        alignment_posteriors, transition_posteriors, log_likelihood, _ = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model, translation_model)
        # Collect statistics in each model.
        prior_model.collect_statistics(len(src_tokens), len(trg_tokens), alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens, alignment_posteriors)
        trasition_model.collect_statistics(len(src_tokens), transition_posteriors)
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood


def estimate_models(src_corpus, trg_corpus, prior_model, translation_model, trasition_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model, trasition_model)
        # M-step
        prior_model.recompute_parameters()
        translation_model.recompute_parameters()
        trasition_model.recompute_parameters()
        if iteration > 0:
            print("corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return prior_model, translation_model, trasition_model


def align_corpus(src_corpus, trg_corpus, prior_model, translation_model, trasition_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        answers = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model, trasition_model)[-1]
        alignments.append(answers)
    return alignments


def initialize_models(src_corpus, trg_corpus):
    prior_model = PriorModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    trasition_model = TransitionModel(src_corpus, trg_corpus)
    return prior_model, translation_model, trasition_model


def normalize(corpus, lemmatize=True):
    lemmatizer = WordNetLemmatizer()
    for sentence in corpus:
        for i, word in enumerate(sentence):
            word = word.lower()
            if lemmatize:
                word = lemmatizer.lemmatize(word)
            sentence[i] = word[:5]
    return corpus


if __name__ == "__main__":
    if not len(sys.argv) == 5:
        print("Usage ./word_alignment.py src_corpus trg_corpus iterations output_prefix.")
        sys.exit(0)
    src_corpus, trg_corpus = read_all_tokens(sys.argv[1]), read_all_tokens(sys.argv[2])
    src_corpus = normalize(src_corpus)
    trg_corpus = normalize(trg_corpus, sys.argv[2].find('lemmas') == -1)
    num_iterations = int(sys.argv[3])
    output_prefix = sys.argv[4]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"
    prior_model, translation_model, trasition_model = initialize_models(src_corpus, trg_corpus)
    prior_model, translation_model, trasition_model = estimate_models(src_corpus, trg_corpus, prior_model, translation_model, trasition_model, num_iterations)
    alignments = align_corpus(src_corpus, trg_corpus, prior_model, translation_model, trasition_model)
    output_alignments_per_test_set(alignments, output_prefix)
