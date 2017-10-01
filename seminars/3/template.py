from collections import Counter, defaultdict
import numpy as np

def train_hmm(tagged_sents):
    """Hidden Markov Model. 
    Calucalte p(tag), p(word|tag), p(tag|tag) from corpus

    Args:
        tagged_sents: list of list of tokens. 
            Example: 
            [('dog', 'N'), ('eats', 'V')]

    Returns:
        p_t, p_w_t, p_t_t - tuple of 3 elements:
        p_t - dict(float), tag->proba
        p_w_t - dict(dict(float), tag -> word -> proba
        p_t_t - dict(dict(float), previous_tag -> tag -> proba
    """
    
    # Your code here

    return p_t, p_w_t, p_t_t


def viterbi_algorithm(test_tokens_list, p_t, p_w_t, p_t_t):
    """Hidden Markov Model. 
    Calucalte p(tag), p(word|tag), p(tag|tag) from corpus

    Args:
        test_tokens_list: list of tokens. 
            Example: 
            ['I', 'go']
        p_t: dict(float), tag->proba
        p_w_t: - dict(dict(float), tag -> word -> proba
        p_t_t: - dict(dict(float), previous_tag -> tag -> proba

    Returns:
        list of hidden tags
    """
    
    # Your code here