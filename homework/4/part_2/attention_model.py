import tensorflow as tf
from keras.layers import Dense, Embedding
import numpy as np

from basic_model import infer_length
from attention_layer import BilinearAttentionLayer


class AttentionBilinearGRUTranslationModel:
    def __init__(self, name, inp_voc,
                 out_voc, emb_size, hid_size):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hid_size = hid_size

        self.attention = BilinearAttentionLayer(self.name, self.hid_size)
        with tf.variable_scope(name):
            self.emb_inp = Embedding(len(inp_voc), emb_size)
            self.emb_out = Embedding(len(out_voc), emb_size)
            self.encoder_forward_GRU = tf.nn.rnn_cell.GRUCell(int(hid_size / 2))  # Forward encoder direction cell
            self.encoder_backward_GRU = tf.nn.rnn_cell.GRUCell(int(hid_size / 2)) # Backward encoder direction cell
            self.decoder_GRU = tf.nn.rnn_cell.GRUCell(hid_size)                   # Decoder cell
            self.logits = Dense(len(out_voc))

            # run on dummy output to .build all layers (and therefore create weights)
            inp = tf.placeholder('int32', [None, None])
            out = tf.placeholder('int32', [None, None])
            encodings, hidden_0 = self.encode(inp)
            attention = self.attention(encodings, hidden_0)
            hidden_1 = self.decode(hidden_0, out[:, 0], attention)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)


    def encode(self, inp, **flags):
        """
        Takes symbolic input sequence, computes initial state and embefings for attention
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp, self.inp_voc.eos_ix)
        inp_emb = self.emb_inp(inp)

        outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
            self.encoder_forward_GRU, self.encoder_backward_GRU,
            inp_emb, sequence_length=inp_lengths, dtype=inp_emb.dtype
        )

        return tf.concat(outputs, 2), tf.concat(final_state, 1)

    def decode(self, prev_state, prev_tokens, attention, **flags):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """
        prev_emb = self.emb_out(prev_tokens[:, None])[:, 0]
        dec_input = tf.concat((prev_emb, attention), axis=1)
        new_output, new_state = self.decoder_GRU(dec_input, prev_state)
        new_output_logits = self.logits(new_output)
        return [new_state], new_output_logits


    def symbolic_score(self, inp, out, eps=1e-30, **flags):
        """
        Takes symbolic int32 matrices of hebrew words and their english translations.
        Computes the log-probabilities of all possible english characters given english prefices and hebrew word.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param out: output sequence, int32 matrix of shape [batch,time]
        :return: log-probabilities of all possible english characters of shape [bath,time,n_tokens]

        NOTE: log-probabilities time axis  is synchronized with out
        In other words, logp are probabilities of __current__ output at each tick, not the next one
        therefore you can get likelihood as logprobas * tf.one_hot(out,n_tokens)
        """
        encodings, first_state = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        def step(blob, y_prev):
            prev_hidden = blob[0]
            attention = self.attention(encodings, prev_hidden)
            new_hidden, logits = self.decode(prev_hidden, y_prev, attention, **flags)
            return list(new_hidden) + [logits]

        results = tf.scan(
            step, initializer=[first_state, first_logits], elems=tf.transpose(out)
        )

        # gather state and logits, each of shape [time, batch, ...]
        logits_seq = results[-1]

        # add initial state and logits
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)

        # convert from [time, batch, ...] to [batch, time, ...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])

        return tf.nn.log_softmax(logits_seq)


    def symbolic_translate(self, inp, greedy=False, max_len=None, eps=1e-30, **flags):
        """
        takes symbolic int32 matrix of hebrew words, produces output tokens sampled
        from the model and output log-probabilities for all possible tokens at each tick.
        :param inp: input sequence, int32 matrix of shape [batch,time]
        :param greedy: if greedy, takes token with highest probablity at each tick.
            Otherwise samples proportionally to probability.
        :param max_len: max length of output, defaults to 2 * input length
        :return: output tokens int32[batch,time] and
                 log-probabilities of all tokens at each tick, [batch,time,n_tokens]
        """
        encodings, first_state = self.encode(inp, **flags)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_ix)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)
        max_len = tf.reduce_max(tf.shape(inp)[1]) * 2

        def step(blob, _):
            prev_hidden, y_prev = blob[0], blob[-1]
            attention = self.attention(encodings, prev_hidden)
            new_hidden, logits = self.decode(prev_hidden, y_prev, attention, **flags)
            y_new = tf.argmax(logits, axis=-1) if greedy else tf.multinomial(logits, 1)[:, 0]
            return list(new_hidden) + [logits, tf.cast(y_new, y_prev.dtype)]

        results = tf.scan(
            step, initializer=[first_state, first_logits, bos], elems=tf.range(max_len)
        )

        # gather state, logits and outs, each of shape [time, batch,...]
        logits_seq, out_seq = results[-2], results[-1]

        # add initial state, logits and out
        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)

        # convert from [time, batch,...] to [batch, time,...]
        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)

        return out_seq, tf.nn.log_softmax(logits_seq)
