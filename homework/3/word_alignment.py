import sys
import numpy as np
from models import PriorModel # <-- Implemented as a uniform distribution.
from models import TranslationModel # <-- Not implemented 
from models import TransitionModel # <-- You will need this for an HMM.
from utils import read_all_tokens, output_alignments_per_test_set
from nltk.stem import WordNetLemmatizer

def get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    alignment_posteriors = np.zeros((len(src_tokens), len(trg_tokens)))
    p_prob = prior_model.get_parameters_for_sentence_pair(len(src_tokens), len(trg_tokens))
    t_prob = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)
    alignment_posteriors += p_prob * t_prob
    alignment_posteriors /= np.sum(p_prob * t_prob, axis=0)

    answers = np.argmax(alignment_posteriors, axis=0)
    arange = np.arange(len(trg_tokens))
    log_likelihood = (
        np.sum(np.log(p_prob[answers, arange])) +
        np.sum(np.log(t_prob[answers, arange]))
    )

    return alignment_posteriors, log_likelihood

def collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # Infer posterior
        alignment_posteriors, log_likelihood = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model)
        # Collect statistics in each model.
        prior_model.collect_statistics(len(src_tokens), len(trg_tokens), alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens, alignment_posteriors)
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood

def estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model)
        # M-step
        prior_model.recompute_parameters()
        translation_model.recompute_parameters()
        if iteration > 0:
            print("corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return prior_model, translation_model

def align_corpus(src_corpus, trg_corpus, prior_model, translation_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        posteriors = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model)[0]
        alignments.append(np.argmax(posteriors, axis=0))
    return alignments

def initialize_models(src_corpus, trg_corpus):
    prior_model = PriorModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    return prior_model, translation_model

def normalize(src_corpus, trg_corpus):
    lemmatizer = WordNetLemmatizer()
    for sentence in trg_corpus:
        for i, word in enumerate(sentence):
            sentence[i] = lemmatizer.lemmatize(word.lower())
    for sentence in src_corpus:
        for i, word in enumerate(sentence):
            sentence[i] = lemmatizer.lemmatize(word.lower())
    return src_corpus, trg_corpus

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage ./word_alignment.py src_corpus trg_corpus iterations output_prefix.")
        sys.exit(0)
    src_corpus, trg_corpus = read_all_tokens(sys.argv[1]), read_all_tokens(sys.argv[2])
    src_corpus, trg_corpus = normalize(src_corpus, trg_corpus)
    num_iterations = int(sys.argv[3])
    output_prefix = sys.argv[4]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"
    prior_model, translation_model = initialize_models(src_corpus, trg_corpus)
    prior_model, translation_model = estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations)    
    alignments = align_corpus(src_corpus, trg_corpus, prior_model, translation_model)
    output_alignments_per_test_set(alignments, output_prefix)
