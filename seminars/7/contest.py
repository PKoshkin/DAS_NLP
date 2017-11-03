import numpy as np


def evaluate_dcg(object_vecs, duplicate_idxs, negative_idxs, k_values):
    """
    Ranks candidates by their embeddings and evaluates the ranking by DCG metric.

    Args:
        object_vecs (ndarray): Embeddings for all objects (questions).
        duplicate_idxs (list([ind1, ind2])): Duplicate indices (as defined by order in object_vecs).
        negative_idxs (list([ind_neg1, ... ind_negN])): Indices of negative objects for each duplicate pair.
        k_values (list): Several sizes of ranked lists for computing DCG@k.

    Returns:

        dcg_values (list): Computed values of DCG_at_k for each k (averaged over examples).
    """

    assert len(duplicate_idxs) == len(negative_idxs)

    # List (by a number of queries) of lists (by a number of different k) of dcg_at_k values. 
    dcg_values = []
 
    for (duplicate_ind1, duplicate_ind2), neg_indxs in zip(duplicate_idxs, negative_idxs):
        candidates = np.hstack([duplicate_ind2, neg_indxs])
        # Compute cosine similarities and sort candidates.
        # Compute DCG for each of k from k_values.
        def cos_dist(vector_1, vector_2):
            return -np.sum(vector_1 * vector_2) / np.sqrt(np.sum(vector_1 ** 2) * np.sum(vector_2 ** 2))
        distance_indexes = np.array([
            cos_dist(object_vecs[candidate], object_vecs[duplicate_ind1])
            for candidate in candidates
        ]).argsort()
        for counter, index in enumerate(distance_indexes):
            if index == 0:
                rank = counter
                break
        rank += 1
        dcg_values.append([
            int(rank <= k) / np.log2(1 + rank)
            for k in k_values
        ])
 
    # Average over different queries.
    dcg_values = np.mean(dcg_values, axis=0)

    return dcg_values


def question2vec_advanced(questions, embeddings):
    """ 
    Computes question embeddings by averaging word embeddings.
    
    Args:
      questions (list of strings): List of questions to be embedded.
      embeddings (gensim object): Pre-trained word embeddings.
      
    Returns:
      ndarray of shape [num_questions, embed_size] with question embeddings.
    """

    result = []
    for question in questions:
        word_embeddings = []
        for word in question.split():
            while True:
                if word == '':
                    break
                if word in embeddings:
                    word_embeddings.append(embeddings[word])
                    break
                else:
                    word = word[:-1]
                    continue
        result.append(np.mean(word_embeddings, axis=0))
    return np.array(result)
