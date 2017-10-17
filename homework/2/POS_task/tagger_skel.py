#!/usr/bin/env python3

import numpy as np
import cityhash
from collections import Counter()

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
TaggedWord = collections.namedtuple('TaggedWord', ['text', 'tag'])
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
            if line[0].isdigit():
                tag = line[3]
                word = line[1]
                tagged_senencies[-1].appned(TaggedWord(text=line[3], tag=line[1]))
            else:
                tagged_senencies.appned([])
        if len(tagged_senencies[-1]) == 0:
            tagged_senencies = tagged_senencies[:-1]
        return tagged_senencies

def write_tagged_sentence(tagged_sentence, f):
    """
    Write tagged sentence to file-like object f.
    """
    file = f.open()
    for tagged_word in tagged_senencies:
        file.write(tagged_word.text + ' ' + tagged_word.tag + '\n')
    file.close()


TaggingQuality = collections.namedtuple('TaggingQuality', ['acc'])


def tagging_quality(ref, out):
    """
    Compute tagging quality and reutrn TaggingQuality object.
    """
    nwords = 0
    ncorrect = 0
    import itertools
    for ref_sentence, out_sentence in itertools.zip_longest(ref, out):
        for ref_word, out_word in itertools.zip_longest():
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
        self._data = np.zeros(n)

    def dot(self, update):
        for position in update._table.keys():
            self._data[position] += update._table[position]

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
            for position x._table.keys():
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


Hypo = collections.namedtuple('Hypo', ['prev', 'pos', 'tagged_word', 'score'])
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


TaggerParams = collections.namedtuple('FeatureParams', [
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
        features = []
        # neighboors words
        for index in range(
            max(hypo.pos - self._tagger_params.src_window, 0),
            min(hypo.pos + self._tagger_params.src_window + 1, len(self._source_sentence))
        ):
            features.appned(h(hypo.tagged_word.tag,
                self._source_sentence[index]
            ) % self._tagger_params.nparams)

        # previous tags
        tmp_hypo = hypo
        while tmp_hypo != None:
            features.appned(h(hypo.tagged_word.tag,
                tmp_hypo.tagged_word.tag
            ) % self._tagger_params.nparams)
            tmp_hypo = tmp_hypo.prev

        # suffixes
        for i in range(1, self._tagger_params.max_suffix):
            features.appned(h(hypo.tagged_word.tag,
                hypo.tagged_word.word[-i:]
            ) % self._tagger_params.nparams)

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
        if hypo is None:
            return [
                Hypo(
                    prev=None,
                    pos=0,
                    tagged_word=TaggedWord(text=self._source_sentence[0], tag=tag),
                    score=0
                )
                for tag in tags
            ]
        else:
            hypos = []
            for tag in tags:
                new_hypo = Hypo(
                    prev=hypo,
                    pos=(hypo.pos + 1),
                    tagged_word=TaggedWord(text=self._source_sentence[hypo.pos + 1], tag=tag),
                    score=0
                )   
                hypos.append(new_hypo._replace(
                    score=self._model.score(self._feature_computer.compute_features(new_hypo))
                ))
            return hypos

    def recombo_hash(self, hypo):
        """
        If two hypos have the same recombination hashes, they can be collapsed
        together, leaving only the hypothesis with a better score.
        """
        return hypo.score


def beam_search(beam_search_task):
    """
    Return list of stacks.
    Each stack contains several hypos, sorted by score in descending 
    order (i.e. better hypos first).
    """
    hypos = sorted(
        beam_search_task.expand(None),
        key=(lambda hypo: hypo.score)
    )[beam_search_task.beam_size():]
    stacks = [hypos]
    for step in range(beam_search_task.total_num_steps() - 1):
        new_hypos = []
        for hypo in hypos:
            new_hypos.expand(beam_search_task.expand(hypo))
        hypos = sorted(
            new_hypos
            key=(lambda hypo: hypo.score)
        )[beam_search_task.beam_size():]
        stacks.appned(hypos)
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
    def __init__(self, ...):
        ...

    def params(self):
        ...

    def loss_and_gradient(self, golden_sentence):
        ...


class StructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        self.tagger_params = tagger_params
        self.model = LinearModel(...)
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
        feature_computer = ...
        for i in range(len(golden_sentence)):
            new_golden_hypo = ...
            golden_hypo = golden_hypo

        # Find where to update.
        golden_head = ...
        rival_head = ...

        # Compute gradient.
        grad = Update()
        while golden_head and rival_head:
            rival_features = feature_computer.compute_features(rival_head)
            grad.assign_madd(self.model.gradient(rival_features, score=None), 1)

            golden_features = feature_computer.compute_features(golden_head)
            grad.assign_madd(self.model.gradient(golden_features, score=None), -1)


            golden_head = golden_head.prev
            rival_head = rival_head.prev

        return grad
        

###############################################################################
#                                                                             #
#                                    SGD                                      #
#                                                                             #
###############################################################################


SGDParams = collections.namedtuple('SGDParams', [
    'epochs',
    'learning_rate',
    'minibatch_size',
    'average' # bool or int
    ])


def make_batches(dataset, minibatch_size):
    """
    Make list of batches from a list of examples.
    """
    ...


def sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn):
    """
    Run (averaged) SGD on a generic optimization task. Modify optimization
    task's parameters.

    After each epoch (and also before and after the whole training),
    run after_each_epoch_fn().
    """
    ...


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
        help='Beam size (0 means unstructured)', type=int, default=1)
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
    optimization_task = optimization_task_cls(...)
    if params is not None:
        print('\n\nLoading parameters from %s\n\n' % cmdargs.model)
        optimization_task.params().assign(params)

    # Validation.
    def after_each_epoch_fn():
        model = LinearModel(cmdargs.nparams)
        model.params().assign(optimization_task.params())
        tagged_sentences = tag_sentences(dataset_dev, ...)
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
        help='test dataset', default='data/en-ud-dev.conllu')
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
        help='Beam size', type=int, default=1)

    return 'test'


def tag_sentences(dataset, ...):
    """
    Tag all sentences in dataset. Dataset is a list of TaggedSentence; while 
    tagging, ignore existing tags.
    """
    ...


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
        nparams=0
        )

    # Load model.
    model = LinearModel(params.values.shape[0])
    model.params().assign(params)

    # Tag all sentences.
    tagged_sentences = tag_sentences(dataset, ...)

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
