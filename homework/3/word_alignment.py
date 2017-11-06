import sys
import numpy as np
from models import PriorModel # <-- Implemented as a uniform distribution.
from models import TranslationModel # <-- Not implemented 
from models import TransitionModel # <-- You will need this for an HMM.
from utils import read_all_tokens, output_alignments_per_test_set
from nltk.stem import WordNetLemmatizer

def get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model, transition_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    alignment_posteriors = np.zeros((len(trg_tokens), len(src_tokens)))
    prior_prob = prior_model.get_parameters_for_sentence_pair(len(src_tokens), len(trg_tokens))
    translation_prob = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)
    transition_prob = transition_model.get_parameters_for_sentence_pair(len(src_tokens))
    alignment_posteriors += prior_prob * translation_prob.T

    transition_posteriors = np.zeros((len(src_tokens), len(src_tokens)))
#    answers = [np.argmax(alignment_posteriors[0])]
#    for i in range(1, len(trg_tokens)):
#        alignment_posteriors[i] *= transition_prob[answers[-1]]
#        answers.append(np.argmax(alignment_posteriors[i]))
#        transition_posteriors[answers[-1]] += alignment_posteriors[i]
#    alignment_posteriors /= np.sum(alignment_posteriors, axis=0)
#    if np.sum(transition_posteriors) != 0:
#        transition_posteriors /= np.sum(transition_posteriors)
#    else:
#        print('bad =(' * 10)
    answers = np.argmax(alignment_posteriors, axis=1)

    arange = np.arange(len(trg_tokens))
    log_likelihood = np.sum(np.log(alignment_posteriors[arange, answers]))

    return alignment_posteriors, transition_posteriors, log_likelihood, answers

def collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model, transition_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # Infer posterior
        alignment_posteriors, transition_posteriors, log_likelihood, _ = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model, transition_model)
        # Collect statistics in each model.
        prior_model.collect_statistics(len(src_tokens), len(trg_tokens), alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens, alignment_posteriors)
        transition_model.collect_statistics(len(src_tokens), transition_posteriors)
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood

def estimate_models(src_corpus, trg_corpus, prior_model, translation_model, transition_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model, transition_model)
        # M-step
        prior_model.recompute_parameters()
        translation_model.recompute_parameters()
        transition_model.recompute_parameters()
        if iteration > 0:
            print("corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return prior_model, translation_model, transition_model

def align_corpus(src_corpus, trg_corpus, prior_model, translation_model, transition_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        answers = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model, transition_model)[3]
        alignments.append(answers)
    return alignments

def initialize_models(src_corpus, trg_corpus):
    prior_model = PriorModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    transition_model = TransitionModel(src_corpus, trg_corpus)
    return prior_model, translation_model, transition_model

def normalize(src_corpus, trg_corpus):
    lemmatizer = WordNetLemmatizer()
    for corpus in [trg_corpus, src_corpus]:
        for sentence in corpus:
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
    prior_model, translation_model, transition_model = initialize_models(src_corpus, trg_corpus)
    prior_model, translation_model, transition_model = estimate_models(src_corpus, trg_corpus, prior_model, translation_model, transition_model, num_iterations)    
    alignments = align_corpus(src_corpus, trg_corpus, prior_model, translation_model, transition_model)
    output_alignments_per_test_set(alignments, output_prefix)
