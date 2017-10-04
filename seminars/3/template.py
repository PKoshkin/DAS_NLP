from collections import Counter, defaultdict
import numpy as np


def train_hmm(tagged_sents):
    """Hidden Markov Model. 
    Calucalte p(tag), p(word|tag), p(tag|tag) from corpus

    Args:
        tagged_sents: list of list of tokens. 
            Example: 
            [[('dog', 'NOUN'), ('eats', 'VERB'), ...], ...]

    Returns:
        p_t, p_w_t, p_t_t - tuple of 3 elements:
        p_t - dict(float), tag->proba
        p_w_t - dict(dict(float)), tag -> word -> proba
        p_t_t - dict(dict(float)), previous_tag -> tag -> proba
    """
    
    # Initialization 
    delta = 1e-24
    counter_tag = Counter()
    counter_tag_tag = Counter()
    counter_tag_word = Counter()
    tags = set()
    words = set()
    for sentence in tagged_sents:
        for word, tag in sentence:
            counter_tag[tag] += 1
            counter_tag_word[(tag, word)] += 1
        for taged, next_taged in zip(sentence, sentence[1:]):
            counter_tag_tag[(taged[1], next_taged[1])] += 1
        tags.update({tag for word, tag in sentence})
        words.update({word for word, tag in sentence})

    p_t_t = defaultdict(lambda: defaultdict(float))
    p_w_t = defaultdict(lambda: defaultdict(float))
    p_t = defaultdict(float)
    # Computing probabilities
    for tag in tags:
        for word in words:
            p_w_t[word][tag] += ((counter_tag_word[(tag, word)] + delta) /
                                 (counter_tag[tag] + delta * len(tags)))
        for next_tag in tags:
            p_t_t[tag][next_tag] += ((counter_tag_tag[(tag, next_tag)] + delta) /
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
    
    tags = np.array([key for key in p_t.keys()])
    delta = [np.array([
        p_w_t[test_tokens_list[0]][k] * p_t[k] for k in tags
    ])]
    s = []
    for i in range(1, len(test_tokens_list)):
        delta.append(np.array([
            np.max([p_w_t[test_tokens_list[i]][k] * p_t_t[m][k] * delta[i - 1][m] for m in range(len(tags))])
            for k in tags
        ]))
        s.append(np.array([
            np.argmax([p_w_t[test_tokens_list[i]][k] * p_t_t[m][k] * delta[i - 1][m] for m in range(len(tags))])
            for k in tags
        ]))
    delta = np.array(delta)
    
    result = [tags[np.argmax(delta[-1, :])]]
    previous_link = np.argmax(delta[-1, :])
    for link in reversed(s):
        result.append(tags[link[previous_link]])
        previous_link = link[previous_link]
    return result[::-1]
