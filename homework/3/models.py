# Models for word alignment.
#
# This file contains stubs for three models to use to model word alignments.
# Notation: i = src_index, I = src_length, j = trg_index, J = trg_length.
# 
# (i) TranslationModel models p(f|e).
# (ii) PriorModel models p(i|j, I, J).
# (iii) TransitionModel models p(a_{j} = i|a_{j-1} = k).
#
# Each model stores parameters (probabilities) and statistics (counts) as has: 
# (i) A method to access a single probability: get_xxx_prob(...).
# (ii) A method to get all probabilities for a sentence pair as a numpy array:
# get_parameters_for_sentence_pair(...).
# (iii) A method to accumulate 'fractional' counts: collect_statistics(...).
# (iv) A method to recompute parameters: recompute_parameters(...).

import numpy as np
from collections import defaultdict

class TranslationModel:
    "Models conditional distribution over trg words given a src word."

    def __init__(self, src_corpus, trg_corpus):
        self._trg_given_src_probs = defaultdict(lambda : defaultdict(lambda : 1.0))
        self._src_trg_counts = defaultdict(lambda : defaultdict(lambda : 0.0))

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token."
        return self._trg_given_src_probs[src_token][trg_token]

    def get_parameters_for_sentence_pair(self, src_tokens, trg_tokens):
        "Return numpy array with t[i][j] = p(f_j|e_i)."
        return np.array([[
                self.get_conditional_prob(src_token, trg_token)
                for trg_token in trg_tokens
            ] for src_token in src_tokens
        ])

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from posterior_matrix[j][i] = p(a_j=i|e, f)"
        for i, src_token in enumerate(src_tokens):
            for j, trg_token in enumerate(trg_tokens):
                self._src_trg_counts[src_token][trg_token] += posterior_matrix[i][j]

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        for src_token in self._src_trg_counts.keys():
            denominamor = np.sum(list(self._src_trg_counts[src_token].values()))
            for trg_token in self._src_trg_counts[src_token].keys():
                prob = self._src_trg_counts[src_token][trg_token] / denominamor
                self._trg_given_src_probs[src_token][trg_token] = prob

        self._src_trg_counts = defaultdict(lambda : defaultdict(lambda : 0.0))


class PriorModel:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for a prior model."
        self._probs = defaultdict(lambda : defaultdict(lambda : 1.0))
        self._counts = defaultdict(lambda : defaultdict(lambda : 0.0))

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a prior probability based on src and trg indices."
        return self._probs[(src_length, trg_length, trg_index)][src_index]
    
    def get_parameters_for_sentence_pair(self, src_length, trg_length):
        "Return a numpy array with all prior p[i][j] = p(i|j, I, J)."
        return np.array([[
                self.get_prior_prob(i, j, src_length, trg_length)
                for j in range(trg_length)
            ] for i in range(src_length)
        ])

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Accumulate counts of alignment events from posterior_matrix[j][i] = p(a_j=i|e, f)"
        for i in range(src_length):
            for j in range(trg_length):
                self._counts[(src_length, trg_length, j)] += posterior_matrix[i][j]

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        for count_key in self._counts.keys():
            denominamor = np.sum(list(self._counts[src_token].values()))
            for i in self._counts[count_key].keys():
                prob = self._counts[count_key][i] / denominamor
                self._probs[count_key][i] = prob

        self._counts = defaultdict(lambda : defaultdict(lambda : 0.0))


class TransitionModel:
    "Models the prior probability of an alignment given the previous token's alignment."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for modeling alignment transitions."
        pass

    def get_parameters_for_sentence_pair(self, src_length):
        "Retrieve the parameters for this sentence pair: A[k, i] = p(a_{j} = i|a_{j-1} = k)"
        pass

    def collect_statistics(self, src_length, transition_posteriors):
        "Accumulate statistics from transition_posteriors[k][i]: p(a_{j} = i, a_{j-1} = k|e, f)"
        pass

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass