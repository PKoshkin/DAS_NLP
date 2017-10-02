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
    """
    
    # Initialization 
    delta = 1e-24
    counter_tag = Counter()
    counter_tag_tag = Counter()
    counter_tag_word = Counter()
    for tag, word in tagged_sents:
        counter_tag[tag] += 1
        counter_tag_word[(tag, word)] += 1
    for tag, next_tag in zip(tagged_sents, tagged_sents[1:]):
        counter_tag_tag[(tag, next_tag)] += 1
    tags = {tag for word, tag in tagged_sents}
    words = {word for word, tag in tagged_sents}
    p_t_t = defaultdict(dict)
    p_w_t = defaultdict(dict)
    p_t = dict()

    # Computing probabilities
    for tag in tags:
        for word in words:
            p_w_t[tag][word] += ((counter_tag_word[(tag, word)] + delta) /
                                 (counter_tag[tag] + delta * len(tags)))
        for next_tag in tags:
            p_w_t[tag][tags] += ((counter_tag_tag[(tag, next_tag)] + delta) /
                                 (counter_tag[tag] + delta * len(tags)))
        p_t[tag] += 1 / len(tags)

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
    
    tags = p_t.keys()
    delta = [np.array([
        p_w_t[(test_tokens_list[0], k)] * p_t[k] for k in tags
    ])]
    s = []
    for i in range(1, len(test_tokens_list)):
        delta.append(np.array([
            np.max([p_w_t[(test_tokens_list[i], k)] * p_t_t[(m, k)] * delta[i - 1][m] for m in tags])
            for tag in tags
        ]))
        s.append(np.array([
            np.argmax([p_w_t[(test_tokens_list[i], k)] * p_t_t[(m, k)] * delta[i - 1][m] for m in tags])
            for tag in tags
        ]))
    delta = np.array(delta)

    result = [np.argmax(delta[-1, :])]
    for link in reversed(s):
        result.append(tokens[link[result[-1]]])
    return result[::-1]
