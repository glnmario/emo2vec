import random
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import normalize
from math import acos, pi, isnan
from scipy.stats import entropy

np.random.seed(13)

NUM_EMOTIONS = 6
NDIMS = 300


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


def clip_to_range_0_1(v):
    v[v <= 0.] = 1e-15
    v[v > 1.] = 1.0


def kl_divergence(true, pred):
    clip_to_range_0_1(true)
    clip_to_range_0_1(pred)
    return entropy(true, pred)


def partition (list, n):
    return [list[i::n] for i in range(n)]

word2idx = {}
line_idx = 0
with open('resources/emotion_specific/bilstm_300d.txt', 'r', encoding='UTF-8') as f:
    next(f)  # skip header
    for line in f:
        values = line.split()

        # probably an error occurred during tokenization
        if len(values) != NDIMS + 1:
            continue

        word = values[0]

        # only one vector for each word
        try:
            word2idx[word]
        except:
            word2idx[word] = line_idx
            line_idx += 1

def build_Y(lexicon, lemma2index, w2i, partition):
    uniform_distr = np.asarray([1 / NUM_EMOTIONS] * NUM_EMOTIONS)
    majority_distr = np.asarray([0,0,0,1,0,0]) + 1e-12
    corpus_prior_distr = np.asarray([1555,761,2815,8240,3830,3849])
    corpus_prior_distr = corpus_prior_distr / np.sum(corpus_prior_distr)

    y_l = np.asarray(DataFrame(lexicon, dtype='float32').T.fillna(0), dtype='float32')
    # y = np.tile(corpus_prior_distr, (len(partition), 1))

    held_out_indices = []
    idx_translator = {}
    y_test = np.zeros((len(w2i), NUM_EMOTIONS))

    i = 0
    for word, idx in lemma2index.items():
        try:
            # if word in corpus
            idx_T = w2i[word]  # get index of word in T

            if idx in partition:
                held_out_indices.append(idx_T)

                y_test[i] = y_l[idx]
                idx_translator[idx_T] = i
                i += 1
        except KeyError:
            continue

    # y = normalize(y, axis=1, norm='l1', copy=False)  # turn multi-labels into prob distribution
    # y_test = normalize(y_test, axis=1, norm='l1', copy=False)

    return uniform_distr, held_out_indices, y_test, idx_translator


_lexicon = dict()
lexicon = dict()
lemma2index = dict()
with open('resources/data/emolex.txt', 'r') as f:
    emo_idx = 0  # anger: 0, anticipation: 1, disgust: 2, fear: 3, joy: 4, sadness: 5, surprise: 6, trust: 7
    i = 0
    for l in f:
        lemma, emotion, has_emotion = read_emo_lemma(l)
        if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
            _lexicon[i] = np.empty(shape=(NUM_EMOTIONS,))
        if emotion in ['positive', 'negative', 'anticipation', 'trust']:
            continue
        _lexicon[i][emo_idx] = has_emotion
        if emo_idx < NUM_EMOTIONS - 1:
            emo_idx += 1
        else:
            norm = np.sum(_lexicon[i])
            if norm != 0:
                lemma2index[lemma] = i
                lexicon[i] = _lexicon[i] / norm
                i += 1

            # reset index - next line contains a new lemma
            emo_idx = 0

print(len(lemma2index))

intersection = [idx for lemma, idx in lemma2index.items() if lemma in word2idx.keys()]
indices = list(intersection)
random.shuffle(indices)


partitions = partition(indices, 10)
kls = []
for p in partitions:
    baseline, heldout, y_test, idx_translator = build_Y(lexicon, lemma2index, word2idx, p)

    divergences = []
    for i in heldout:
        i_ = idx_translator[i]

        divergences.append(kl_divergence(y_test[i_], baseline))

    kl = np.sum(np.asarray(divergences)) / len(divergences)
    kls.append(kl)
    print(kl)

result = np.sum(np.asarray(kls)) / len(kls)
print('Average kl:', result)

# remove 0 vectors from both y_test and the baseline
