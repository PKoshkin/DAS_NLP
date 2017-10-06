#!/usr/bin/env python3


###############################################################################
#                                                                             #
#                                INPUT DATA                                   #
#                                                                             #
###############################################################################


def read_tags(path):
        """
    Read a list of possible tags from file and return the list.
    """
    ...


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
    ...

def write_tagged_sentence(tagged_sentence, f):
    """
    Write tagged sentence to file-like object f.
    """
    ...


TaggingQuality = collections.namedtuple('TaggingQuality', ['acc'])


def tagging_quality(ref, out):
    """
    Compute tagging quality and reutrn TaggingQuality object.
    """
    ...


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
        ...

    def dot(self, update):
        ...

    def assign(self, other):
        ...

    def assign_mul(self, coeff):
        ...

    def assign_madd(self, x, coeff):
        """
        self = self + x * coeff
        x can be either Value or Update.
        """
        ...


class Update:
    """
    Sparse object that holds an update of parameters.
    """

    def __init__(self, hashes=None, values=None):
        ...

    def assign_mul(self, coeff):
        ...

    def assign_madd(self, update, coeff):
        ...


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
        ...

    def params(self):
        ...

    def score(self, features):
        ...

    def gradient(self, features, score):
        ...


###############################################################################
#                                                                             #
#                                    HYPO                                     #
#                                                                             #
###############################################################################


Hypo = collections.namedtuple('Hypo', ['prev', 'pos', 'tagged_word', 'score'])


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
    def __init__(self, ...):
        ...

    def compute_features(self, hypo):
        """
        Compute features for a given Hypo and return Update.
        """
        ...


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

    def __init__(self, ...):
        ...

    def num_stacks(self):
        """
        Return total number of stacks.
        """
        ...

    def beam_size(self):
        ...

    def expand(self, hypo):
        """
        Given Hypo, return a list of its possible expansions.
        'hypo' might be None -- return a list of initial hypos then.
        """
        ...

    def recombo_hash(self, hypo):
        """
        If two hypos have the same recombination hashes, they can be collapsed
        together, leaving only the hypothesis with a better score.
        """
        ...


def beam_search(beam_search_task, feature_computer, model):
    """
    Return list of stacks.
    Each stack contains several hypos, sorted by score in descending 
    order (i.e. better hypos first).
    """
    ...


def make_tag_lattice(golden_sentence, tags):
    """
    Create a tag lattice from words of original sentence.
    Ignore tags in golden_sentence.
    """
    ...


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
    def __init__(self, ...):
        ...

    def params(self):
        ...

    def loss_and_gradient(self, golden_sentence):
        ...


###############################################################################
#                                                                             #
#                                    SGD                                      #
#                                                                             #
###############################################################################


SGDParams = collections.namedtuple('SGDParams', [
    'epochs',
    'learning_rate',
    'minibatch_size',
    'average'
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
