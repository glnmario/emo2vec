from __future__ import print_function

import random

import numpy as np
from pandas import DataFrame
from scipy.stats.stats import _sum_of_squares
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from math import acos, pi, isnan
# from scipy.stats import pearsonr, spearmanr

np.random.seed(13)

NUM_EMOTIONS = 6
NDIMS = 300


def pearsonr(x, y):
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    clip_to_range_0_1(x)
    clip_to_range_0_1(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x - mx, y - my
    clip_to_range_0_1(xm)
    clip_to_range_0_1(ym)
    r_num = np.add.reduce(xm * ym)
    r_den = np.sqrt(_sum_of_squares(xm) * _sum_of_squares(ym))
    r = r_num / r_den

    # Presumably, if abs(r) > 1, then it is only some small artifact of floating
    # point arithmetic.
    return max(min(r, 1.0), -1.0)


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


def to_cosine_dist(similarity):
    """
    Convert cosine similarity (in range [-1,1]) to cosine distance (in range [0,1])
    :param similarity: cosine similarity
    :return: cosine distance
    """
    if similarity < -1:
        similarity = -1
    if similarity > 1:
        similarity = 1

    return acos(similarity) / pi


def clip_to_range_0_1(v):
    v[v <= 0.] = 1e-15
    v[v > 1.] = 1.0


def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    id = np.eye(a, a)
    return np.linalg.lstsq(m, id)[0]


def partition (list, n):
    return [list[i::n] for i in range(n)]


def build_Y(lexicon, lemma2index, word2index_T, partition):
    """
    :param lexicon_path: emotion lexicon
    :param word2index_T: dictionary word -> index referring to T matrix
    :return: (Y matrix, indices of labeled instances)
    """

    y_l = np.asarray(DataFrame(lexicon, dtype='float16').T.fillna(0), dtype='float16')
    y = np.random.random((len(word2index_T), NUM_EMOTIONS))

    labeled_indices = []
    held_out_indices = []
    idx_translator = {}
    y_test = np.empty((len(partition), NUM_EMOTIONS))

    i = 0
    for word, idx in lemma2index.items():
        try:
            # if word in corpus
            idx_T = word2index_T[word]  # get index of word in T

            if idx in partition:
                held_out_indices.append(idx_T)
                y_test[i] = y_l[idx]
                idx_translator[idx_T] = i
                i += 1
            else:
                labeled_indices.append(idx_T)
                y[idx_T] = y_l[idx]  # set values of labeled word
        except KeyError:
            continue

    y = normalize(y, axis=1, norm='l1', copy=False)  # turn multi-labels into prob distribution
    y_test = normalize(y_test, axis=1, norm='l1', copy=False)

    return y, labeled_indices, held_out_indices, y_test, idx_translator


def build_T(model_path, sigma):
    """
    :param model_path: the path of the final model
    :return: (T matrix, word-to-index dictionary for T)
    """
    emo2vec = KeyedVectors.load_word2vec_format(model_path, binary=False)  # .init_sims(replace=True)
    idx2word = dict(enumerate(emo2vec.index2word))
    n = 100#len(idx2word)

    # invert idx -> word mapping
    # word2idx = {w: i for i, w in idx2word.items()}
    word2idx = {w: i for i, w in {i: idx2word[i] for i in np.arange(n)}.items()}

    t = np.empty((n, n), dtype='float16')

    for w1, i in word2idx.items():
        for w2, j in word2idx.items():
            t[i, j] = to_cosine_dist(emo2vec.similarity(w1, w2)) ** 2

    t /= sigma ** 2
    t = np.exp(-t)
    t = normalize(t, axis=0, norm='l1', copy=False)
    t = normalize(t, axis=1, norm='l1', copy=False)

    return t, word2idx


def labelprop(model, lexicon_path, sigma):
    print('Build T...')
    T, word2idx_T = build_T(model, sigma)


    lexicon = dict()
    lemma2index = dict()

    with open(lexicon_path, 'r') as f:
        emo_idx = 0  # anger: 0, anticipation: 1, disgust: 2, fear: 3, joy: 4, sadness: 5, surprise: 6, trust: 7
        i = 0
        for l in f:
            lemma, emotion, has_emotion = read_emo_lemma(l)
            if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
                lexicon[i] = np.empty(shape=(NUM_EMOTIONS,))
                lemma2index[lemma] = i
            if emotion in ['positive', 'negative', 'anticipation', 'trust']:
                continue
            lexicon[i][emo_idx] = has_emotion
            if emo_idx < NUM_EMOTIONS - 1:
                emo_idx += 1
            else:
                # reset index - next line contains a new lemma
                emo_idx = 0
                i += 1

    intersection = [idx for lemma, idx in lemma2index.items() if lemma in word2idx_T.keys()]
    indices = list(intersection)
    random.shuffle(indices)

    partitions = partition(indices, 10)

    rs = []
    for p in partitions:
        Y, L, heldout, y_test, idx_translator = build_Y(lexicon, lemma2index, word2idx_T, p)

        U = np.setdiff1d(np.asarray(list(word2idx_T.values()), dtype='int32'), L)  # unlabeled nodes indices

        T_uu = T[U][:, U]
        T_ul = T[U][:, L]

        n, m = T_uu.shape
        I = np.eye(n, m)

        clip_to_range_0_1(Y)

        Y[U] = inv(I - T_uu) @ T_ul @ Y[L]

        clip_to_range_0_1(Y)
        Y = normalize(Y, axis=1, norm='l1', copy=False)

        correlations = []
        for i in heldout:
            i_ = idx_translator[i]
            correlations.append(pearsonr(Y[i], y_test[i_]))

        r = np.sum(np.asarray(correlations)) / len(correlations)
        rs.append(r)

    for j in range(len(partitions)):
        print(j, ':', rs[j])

    result = np.sum(np.asarray(rs)) / len(rs)
    # with open('crossval-sigma01', 'w') as f:
    print('Average r:', result)


sigma = 0.1

labelprop('resources/emotion_specific/bilstm_300d.txt', 'resources/data/emolex.txt', sigma)

