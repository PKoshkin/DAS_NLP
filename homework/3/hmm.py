# Example HMM implementation 
#
# The python code is shamelessly stolen from Trevor Cohn.
# Original: http://people.eng.unimelb.edu.au/tcohn/comp90042/HMM.py
#
# This is a generic and very readable implementation of a discrete HMM but it
# will need adapting to do word alignment.
#
# 1. This implementation assumes a fixed number of hidden states across all sequences.
# In word alignment, however, the number of hidden states is equal to the number of
# source tokens so your HMM parameters (pi, O, A) will change for each sentence. 
#
# 2. This implementation is very readable but not very efficient or practical.
# (i) You should make it more efficent by collapsing loops into numpy matrix operations.
# (ii) You should scale the forward and backward probabilities (i.e. alpha and beta) to
# avoid numerical underflow on longer sequences.
#
# 3. You will probably only need the forward and backward methods from this file.
# Once you have the alpha and beta probabilities you can easily compute the statistics
# needed by the models in models.py.

import numpy as np

def forward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    
    alpha = np.zeros((N, S))
    
    # base case
    alpha[0, :] = pi * O[:,observations[0]]
    
    # recursive case
    for i in range(1, N):
        for s2 in range(S):
            for s1 in range(S):
                alpha[i, s2] += alpha[i-1, s1] * A[s1, s2] * O[s2, observations[i]]
    
    return (alpha, np.sum(alpha[N-1,:]))


def backward(params, observations):
    pi, A, O = params
    N = len(observations)
    S = pi.shape[0]
    
    beta = np.zeros((N, S))
    
    # base case
    beta[N-1, :] = 1
    
    # recursive case
    for i in range(N-2, -1, -1):
        for s1 in range(S):
            for s2 in range(S):
                beta[i, s1] += beta[i+1, s2] * A[s1, s2] * O[s2, observations[i+1]]
    
    return (beta, np.sum(pi * O[:, observations[0]] * beta[0,:]))


def baum_welch(training, pi, A, O, iterations):
    pi, A, O = np.copy(pi), np.copy(A), np.copy(O) # take copies, as we modify them
    S = pi.shape[0]
    
    # do several steps of EM hill climbing
    for it in range(iterations):
        pi1 = np.zeros_like(pi)
        A1 = np.zeros_like(A)
        O1 = np.zeros_like(O)
        
        for observations in training:
            # compute forward-backward matrices
            alpha, za = forward((pi, A, O), observations)
            beta, zb = backward((pi, A, O), observations)
            assert abs(za - zb) < 1e-6, "it's badness 10000 if the marginals don't agree"
            
            # M-step here, calculating the frequency of starting state, transitions and (state, obs) pairs
            pi1 += alpha[0,:] * beta[0,:] / za
            for i in range(0, len(observations)):
                O1[:, observations[i]] += alpha[i,:] * beta[i,:] / za
            for i in range(1, len(observations)):
                for s1 in range(S):
                    for s2 in range(S):
                        A1[s1, s2] += alpha[i-1,s1] * A[s1, s2] * O[s2, observations[i]] * beta[i,s2] / za
                                                                    
        # normalise pi1, A1, O1
        pi = pi1 / np.sum(pi1)
        for s in range(S):
            A[s, :] = A1[s, :] / np.sum(A1[s, :])
            O[s, :] = O1[s, :] / np.sum(O1[s, :])
        
        print pi, A, O
    
    return pi, A, O
