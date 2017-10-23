#!/usr/bin/env python3

import numpy as np
from math import ceil
from random import shuffle
import cityhash
import argparse
import os
import sys
import re
import pprint
import pickle
from collections import Counter
from collections import namedtuple

###############################################################################
#                                                                             #
#                                INPUT DATA                                   #
#                                                                             #
###############################################################################


def read_tags(path):
    """
    Read a list of possible tags from file and return the list.
    """
    with open(path) as file:
        return [line[:-1] for line in file]


# Word: str
# Sentence: list of str
TaggedWord = namedtuple('TaggedWord', ['text', 'tag'])
# TaggedSentence: list of TaggedWord
# Tags: list of TaggedWord
# TagLattice: list of Tags


def read_tagged_sentences(path):
    """
    Read tagged sentences from file and return array of TaggedSentence.
    """
    with open(path) as file:
        tagged_senencies = [[]]
        for line in file:
            line = line.split()
            if len(line) == 0:
                continue
            if line[0].isdigit():
                tagged_senencies[-1].append(TaggedWord(text=line[1], tag=line[3]))
            else:
                if len(tagged_senencies) != 0:
                    if len(tagged_senencies[-1]) != 0:
                        tagged_senencies.append([])
        if len(tagged_senencies[-1]) == 0:
            tagged_senencies = tagged_senencies[:-1]
        return tagged_senencies

def write_tagged_sentence(tagged_sentence, f):
    """
    Write tagged sentence to file-like object f.
    """
    for i, tagged_word in enumerate(tagged_sentence):
        f.write(str(i + 1) + '\t' + tagged_word.text + '\t\t' + tagged_word.tag + '\t\t\t\t\t\t\n')
    print('\n')


TaggingQuality = namedtuple('TaggingQuality', ['acc'])


def tagging_quality(ref, out):
    """
    Compute tagging quality and reutrn TaggingQuality object.
    """
    nwords = 0
    ncorrect = 0
    import itertools
    for ref_sentence, out_sentence in itertools.zip_longest(ref, out):
        for ref_word, out_word in itertools.zip_longest(ref_sentence, out_sentence):
            nwords += 1
            if ref_word.tag == out_word.tag:
                ncorrect += 1
    return TaggingQuality(acc=(ncorrect / nwords))


###############################################################################
#                                                                             #
#                             VALUE & UPDATE                                  #
#                                                                             #
###############################################################################


class Value:
    """
    Dense object that holds parameters.
    """

    def __init__(self, n):
        self._data = np.ones(n)

    def dot(self, update):
        result = 0
        for position in update._table.keys():
            result += self._data[position] * update._table[position]
        return result

    def assign(self, other):
        """
        self = other
        other is Value.
        """
        np.copyto(self._data, other._data)

    def assign_mul(self, coeff):
        """
        self = self * coeff
        coeff is float.
        """
        self._data *= coeff

    def assign_madd(self, x, coeff):
        """
        self = self + x * coeff
        x can be either Value or Update.
        coeff is float.
        """
        if isinstance(x, Value):
            self._data += x._data * coeff
        else:
            for position in x._table.keys():
                self._data[position] += x._table[position] * coeff


class Update:
    """
    Sparse object that holds an update of parameters.
    """

    def __init__(self, positions=None, values=None):
        """
        positions: array of int
        values: array of float
        """
        self._table = Counter()
        if positions is not None:
            for position, value in zip(positions, values):
                self._table[position] += value

    def assign_mul(self, coeff):
        """
        self = self * coeff
        coeff: float
        """
        for key in self._table.keys():
            self._table[key] *= coeff

    def assign_madd(self, update, coeff):
        """
        self = self + update * coeff
        coeff: float
        """
        for position in update._table.keys():
            self._table[position] += coeff * update._table[position]

###############################################################################
#                                                                             #
#                                  MODEL                                      #
#                                                                             #
###############################################################################


Features = Update


class LinearModel:
    """
    A thing that computes score and gradient for given features.
    """

    def __init__(self, n):
        self._params = Value(n)

    def params(self):
        return self._params

    def score(self, features):
        """
        features: Update
        """
        return self._params.dot(features)

    def gradient(self, features, score):
        return features


###############################################################################
#                                                                             #
#                                    HYPO                                     #
#                                                                             #
###############################################################################


Hypo = namedtuple('Hypo', ['prev', 'pos', 'tagged_word', 'score'])
# prev: previous Hypo
# pos: position of word (0-based)
# tagged_word: tagging of source_sentence[pos]
# score: sum of scores over edges

###############################################################################
#                                                                             #
#                              FEATURE COMPUTER                               #
#                                                                             #
###############################################################################


def h(x):
    """
    Compute CityHash of any object.
    Can be used to construct features.
    """
    return cityhash.CityHash64(repr(x))


TaggerParams = namedtuple('FeatureParams', [
    'src_window',
    'dst_order',
    'max_suffix',
    'beam_size',
    'nparams'
])


class FeatureComputer:
    def __init__(self, tagger_params, source_sentence):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence


    def compute_features(self, hypo):
        """
        Compute features for a given Hypo and return Update.
        """
        def make_feature(feature, feature_name):
            return h((feature_name, hypo.tagged_word.tag, feature)) % self._tagger_params.nparams 
        regulars = [re.compile(r'[A-Z]'), re.compile(r'[0-9]'), re.compile(r'-')]
        features = [
            make_feature((regular.search(hypo.tagged_word.text) is None), i)
            for i, regular in enumerate(regulars)
        ]
        feature_counter = len(features)

        # neighboors words
        min_index = max(hypo.pos - self._tagger_params.src_window, 0)
        max_index = min(hypo.pos + self._tagger_params.src_window + 1, len(self._source_sentence))
        for index in range(min_index, max_index):
            features.append(
                make_feature(self._source_sentence[index], feature_counter)
            )
            feature_counter += 1

        # previous tags
        tmp_hypo = hypo
        tags = tuple()
        counter = 0
        while tmp_hypo != None:
            tags += (tmp_hypo.tagged_word.tag, )
            features.append(
                make_feature(tags, feature_counter)
            )
            feature_counter += 1
            tmp_hypo = tmp_hypo.prev
            counter += 1
            if counter > self._tagger_params.dst_order:
                break

        # suffixes
        for i in range(1, self._tagger_params.max_suffix + 1):
            features.append(
                make_feature(hypo.tagged_word.text[-i:], feature_counter)
            )
            feature_counter += 1

        # prefixes
        for i in range(1, self._tagger_params.max_suffix + 1):
            features.append(
                make_feature(hypo.tagged_word.text[:i], feature_counter)
            )
            feature_counter += 1

        return Update(positions=features, values=np.ones(len(features)))

###############################################################################
#                                                                             #
#                                BEAM SEARCH                                  #
#                                                                             #
###############################################################################


class BeamSearchTask:
    """
    An abstract beam search task. Can be used with beam_search() generic 
    function.
    """

    def __init__(self, tagger_params, source_sentence, model, tags):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence
        self._model = model
        self._tags = tags
        self._feature_computer = FeatureComputer(tagger_params, source_sentence)

    def total_num_steps(self):
        """
        Number of hypotheses between beginning and end (number of words in
        the sentence).
        """
        return len(self._source_sentence)

    def beam_size(self):
        return self._tagger_params.beam_size

    def expand(self, hypo):
        """
        Given Hypo, return a list of its possible expansions.
        'hypo' might be None -- return a list of initial hypos then.

        Compute hypotheses' scores inside this function!
        """
        hypos = []
        position = (hypo.pos + 1) if hypo is not None else 0
        for tag in self._tags:
            new_hypo = Hypo(
                prev=hypo,
                pos=position,
                tagged_word=TaggedWord(text=self._source_sentence[position], tag=tag),
                score=0
            )
            hypos.append(new_hypo._replace(
                score=self._model.score(self._feature_computer.compute_features(new_hypo)) +
                (hypo.score if hypo is not None else 0)
            ))
        return hypos

    def recombo_hash(self, hypo):
        """
        If two hypos have the same recombination hashes, they can be collapsed
        together, leaving only the hypothesis with a better score.
        """
        return None


def beam_search(beam_search_task):
    """
    Return list of stacks.
    Each stack contains several hypos, sorted by score in descending 
    order (i.e. better hypos first).
    """
    hypos = sorted(
        beam_search_task.expand(None),
        key=(lambda hypo: hypo.score)
    )[-beam_search_task.beam_size():]
    stacks = [hypos]
    for step in range(beam_search_task.total_num_steps() - 1):
        new_hypos = []
        for hypo in hypos:
            new_hypos.extend(beam_search_task.expand(hypo))
        hypos = sorted(
            new_hypos,
            key=(lambda hypo: hypo.score)
        )[-beam_search_task.beam_size():]
        stacks.append(hypos)
    return stacks
        

###############################################################################
#                                                                             #
#                            OPTIMIZATION TASKS                               #
#                                                                             #
###############################################################################


class OptimizationTask:
    """
    Optimization task that can be used with sgd().
    """

    def params(self):
        """
        Parameters which are optimized in this optimization task.
        Return Value.
        """
        raise NotImplementedError()

    def loss_and_gradient(self, golden_sentence):
        """
        Return (loss, gradient) on a specific example.

        loss: float
        gradient: Update
        """
        raise NotImplementedError()


class UnstructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        raise NotImplementedError()

    def params(self):
        raise NotImplementedError()

    def loss_and_gradient(self, golden_sentence):
        raise NotImplementedError()


class StructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        self.tagger_params = tagger_params
        self.model = LinearModel(tagger_params.nparams)
        self.tags = tags

    def params(self):
        return self.model.params()

    def loss_and_gradient(self, golden_sentence):
        # Do beam search.
        beam_search_task = BeamSearchTask(
            self.tagger_params, 
            [golden_tagged_word.text for golden_tagged_word in golden_sentence], 
            self.model, 
            self.tags
        )
        stacks = beam_search(beam_search_task)

        # Compute chain of golden hypos (and their scores!).
        golden_hypo = None
        feature_computer = FeatureComputer(
            self.tagger_params,
            [tagged_word.text for tagged_word in golden_sentence]
        )
        max_violation = 0
        for i, word in enumerate(golden_sentence):
            new_golden_hypo = Hypo(
                prev=golden_hypo,
                pos=i,
                tagged_word=word,
                score=0
            )
            golden_hypo = new_golden_hypo._replace(score=self.model.score(
                feature_computer.compute_features(
                    new_golden_hypo
                )) + (golden_hypo.score if golden_hypo is not None else 0)
            )
            new_violation = stacks[i][0].score - golden_hypo.score
            if new_violation > max_violation:
                golden_head = golden_hypo
                rival_head = stacks[i][-1]
                max_violation = new_violation

        if max_violation == 0:
            rival_head = stacks[-1][-1]
            golden_head = golden_hypo


        # Compute gradient.
        grad = Update()
        while golden_head and rival_head:
            rival_features = feature_computer.compute_features(rival_head)
            grad.assign_madd(self.model.gradient(rival_features, score=None), 1)

            golden_features = feature_computer.compute_features(golden_head)
            grad.assign_madd(self.model.gradient(golden_features, score=None), -1)

            golden_head = golden_head.prev
            rival_head = rival_head.prev

        return None, grad

###############################################################################
#                                                                             #
#                                    SGD                                      #
#                                                                             #
###############################################################################


SGDParams = namedtuple('SGDParams', [
    'epochs',
    'learning_rate',
    'minibatch_size',
    'average' # bool or int
])


def make_batches(dataset, minibatch_size):
    """
    Make list of batches from a list of examples.
    """
    minibatches_number = ceil(len(dataset) / minibatch_size)
    result = [
        dataset[(i * minibatch_size):((i + 1) * minibatch_size)]
        for i in range(minibatches_number)
    ]
    shuffle(result)
    return result


def sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn):
    """
    Run (averaged) SGD on a generic optimization task. Modify optimization
    task's parameters.

    After each epoch (and also before and after the whole training),
    run after_each_epoch_fn().
    """
    batches = make_batches(dataset, sgd_params.minibatch_size)
    params_sum = Value(optimization_task.tagger_params.nparams)
    sum_length = 0
    after_each_epoch_fn()
    for epoch in range(sgd_params.epochs):
        for i, batch in enumerate(batches):
            grad = Update()
            for example in batch:
                new_grad = optimization_task.loss_and_gradient(example)[1]
                grad.assign_madd(new_grad, 1)
            optimization_task.params().assign_madd(grad, -sgd_params.learning_rate)
            if i % sgd_params.average == 0:
                params_sum.assign_madd(optimization_task.params(), 1)
                sum_length += 1
        after_each_epoch_fn()
    params_sum.assign_mul(1 / sum_length)
    optimization_task.params().assign(params_sum)
    after_each_epoch_fn()

###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


# - Train - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def TRAIN_add_cmdargs(subp):
    p = subp.add_parser('train')

    p.add_argument('--tags',
        help='tags file', type=str, default='data/tags')
    p.add_argument('--dataset',
        help='train dataset', default='data/en-ud-train.conllu')
    p.add_argument('--dataset-dev',
        help='dev dataset', default='data/en-ud-dev.conllu')
    p.add_argument('--model',
        help='NPZ model', type=str, default='model.npz')
    p.add_argument('--sgd-epochs',
        help='SGD number of epochs', type=int, default=15)
    p.add_argument('--sgd-learning-rate',
        help='SGD learning rate', type=float, default=0.01)
    p.add_argument('--sgd-minibatch-size',
        help='SGD minibatch size (in sentences)', type=int, default=32)
    p.add_argument('--sgd-average',
        help='SGD average every N batches', type=int, default=32)
    p.add_argument('--tagger-src-window',
        help='Number of context words in input sentence to use for features',
        type=int, default=2)
    p.add_argument('--tagger-dst-order',
        help='Number of context tags in output tagging to use for features',
        type=int, default=3)
    p.add_argument('--tagger-max-suffix',
        help='Maximal number of prefix/suffix letters to use for features',
        type=int, default=4)
    p.add_argument('--beam-size',
        help='Beam size (0 means unstructured)', type=int, default=5)
    p.add_argument('--nparams',
        help='Parameter vector size', type=int, default=2**22)

    return 'train'

def TRAIN(cmdargs):
    # Beam size.
    optimization_task_cls = StructuredPerceptronOptimizationTask
    if cmdargs.beam_size == 0:
        cmdargs.beam_size = 1
        optimization_task_cls = UnstructuredPerceptronOptimizationTask

    # Parse cmdargs.
    tags = read_tags(cmdargs.tags)
    dataset = read_tagged_sentences(cmdargs.dataset)
    dataset_dev = read_tagged_sentences(cmdargs.dataset_dev)
    params = None
    if os.path.exists(cmdargs.model):
        params = pickle.load(open(cmdargs.model, 'rb'))
    sgd_params = SGDParams(
        epochs=cmdargs.sgd_epochs,
        learning_rate=cmdargs.sgd_learning_rate,
        minibatch_size=cmdargs.sgd_minibatch_size,
        average=cmdargs.sgd_average
    )
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=cmdargs.nparams
    )

    # Load optimization task
    optimization_task = optimization_task_cls(tagger_params, tags)
    if params is not None:
        print('\n\nLoading parameters from %s\n\n' % cmdargs.model)
        optimization_task.params().assign(params)

    # Validation.
    def after_each_epoch_fn():
        model = LinearModel(cmdargs.nparams)
        model.params().assign(optimization_task.params())
        tagged_sentences = tag_sentences(dataset_dev, tagger_params, model, tags)
        q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset_dev))
        print()
        print(q)
        print()

        # Save parameters.
        print('\n\nSaving parameters to %s\n\n' % cmdargs.model)
        pickle.dump(optimization_task.params(), open(cmdargs.model, 'wb'))

    # Run SGD.
    sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn)


# - Test  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def TEST_add_cmdargs(subp):
    p = subp.add_parser('test')

    p.add_argument('--tags',
        help='tags file', type=str, default='data/tags')
    p.add_argument('--dataset',
        help='test dataset', default='data/en-ud-test-notags.conllu')
    p.add_argument('--model',
        help='NPZ model', type=str, default='model.npz')
    p.add_argument('--tagger-src-window',
        help='Number of context words in input sentence to use for features',
        type=int, default=2)
    p.add_argument('--tagger-dst-order',
        help='Number of context tags in output tagging to use for features',
        type=int, default=3)
    p.add_argument('--tagger-max-suffix',
        help='Maximal number of prefix/suffix letters to use for features',
        type=int, default=4)
    p.add_argument('--beam-size',
        help='Beam size', type=int, default=5)

    return 'test'


def tag_sentences(dataset, tagger_params, model, tags):
    """
    Tag all sentences in dataset. Dataset is a list of TaggedSentence; while 
    tagging, ignore existing tags.
    """
    tagged_dataset = []
    for i, sentence in enumerate(dataset):
        beam_search_task = BeamSearchTask(
            tagger_params,
            [tagged_word.text for tagged_word in sentence],
            model,
            tags
        )
        hypo = beam_search(beam_search_task)[-1][-1]
        tagged_sentence = []
        while hypo is not None:
            tagged_sentence.append(hypo.tagged_word)
            hypo = hypo.prev
        tagged_sentence = list(reversed(tagged_sentence))
        tagged_dataset.append(tagged_sentence)
    return tagged_dataset

def TEST(cmdargs):
    # Parse cmdargs.
    tags = read_tags(cmdargs.tags)
    dataset = read_tagged_sentences(cmdargs.dataset)
    params = pickle.load(open(cmdargs.model, 'rb'))
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=len(params._data)
    )

    # Load model.
    model = LinearModel(np.shape(params._data))
    model.params().assign(params)

    # Tag all sentences.
    tagged_sentences = tag_sentences(dataset, tagger_params, model, tags)

    # Write tagged sentences.
    for tagged_sentence in tagged_sentences:
        write_tagged_sentence(tagged_sentence, sys.stdout)

    # Measure and print quality.
    q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset))
    print(q, file=sys.stderr)


# - Main  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def main():
    # Create parser.
    p = argparse.ArgumentParser('tagger.py')
    subp = p.add_subparsers(dest='cmd')

    # Add subcommands.
    train = TRAIN_add_cmdargs(subp)
    test = TEST_add_cmdargs(subp)

    # Parse.
    cmdargs = p.parse_args()

    # Run.
    if cmdargs.cmd == train:
        TRAIN(cmdargs)
    elif cmdargs.cmd == test:
        TEST(cmdargs)
    else:
        p.error('No command')

if __name__ == '__main__':
    main()
