from collections import Counter
import re

def get_words_counter(corpus):
    corpus = re.sub("n't", ' not', corpus)
    corpus = re.sub("'s", ' is', corpus)
    words = re.findall('[a-zA-Z]+', corpus)
    words_lower = [word.lower() for word in words]
    word_lower_counts = Counter(words_lower)
    return word_lower_counts
