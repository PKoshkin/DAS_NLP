import tensorflow as tf
import numpy as np


class BilinearAttentionLayer:
    def __init__(self, name, hid_size):
        """
        A basic layer that computes attention weights and response
        Uses bilinear function.
        """
        self.name = name
        self.hid_size = hid_size

        with tf.variable_scope(name):
            self.attention_W = tf.Variable(np.zeros((self.hid_size, self.hid_size)), dtype=np.float32)

    def __call__(self, encodings, prev_hidden):
        """
        Takes encodings and previous hidden state as input.
        Returns attention vector with bilinear function.
        """
        with tf.variable_scope(self.name):
            scores = tf.tensordot(encodings, self.attention_W, axes=[[2], [0]])
            scores = tf.transpose(scores, [1, 0, 2])
            scores = tf.reduce_sum(scores * prev_hidden, axis=2)
            probs = tf.nn.softmax(scores, dim=0)
            attention = tf.reduce_sum(probs * tf.transpose(encodings, [2, 1, 0]), axis=1)
            attention = tf.transpose(attention, [1, 0])

            return attention
