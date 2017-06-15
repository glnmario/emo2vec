from __future__ import print_function
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from math import acos, pi

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


def build_Y(lexicon_path, word2index_T):
    """
    :param lexicon_path: emotion lexicon
    :param word2index_T: dictionary word -> index referring to T matrix
    :return: (Y matrix, indices of labeled instances)
    """
    lexicon = dict()
    lexeme2index = dict()

    with open(lexicon_path, 'r') as f:
        emo_idx = 0  # anger: 0, disgust: 1, fear: 2, joy: 3, sadness: 4, surprise: 6
        i = 0
        for l in f:
            lemma, emotion, has_emotion = read_emo_lemma(l)
            if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
                lexicon[i] = np.empty(shape=(NUM_EMOTIONS,))
                lexeme2index[lemma] = i
            if emotion in ['positive', 'negative', 'anticipation', 'trust']:
                continue
            lexicon[i][emo_idx] = has_emotion
            if emo_idx < NUM_EMOTIONS - 1:
                emo_idx += 1
            else:
                # reset index - next line contains a new lemma
                emo_idx = 0
                i += 1

    y_l = np.asarray(
        DataFrame(lexicon, dtype='float16').T.fillna(0), dtype='float16')
    y = np.random.random((len(word2index_T), NUM_EMOTIONS))

    labeled_indices = []
    for word, idx in lexeme2index.items():
        try:
            # if word in corpus
            idx_T = word2index_T[word]  # get index of word in T
            y[idx_T] = y_l[idx]  # set values of labeled word
            labeled_indices.append(idx_T)
        except KeyError:
            continue

    # turn multi-labels into prob distribution
    y = normalize(y, axis=1, norm='l1', copy=False)

    return y, labeled_indices


def build_T(model_path, sigma):
    """
    :param model_path: the path of the final model
    :return: (T matrix, word-to-index dictionary for T)
    """
    emo2vec = KeyedVectors.load_word2vec_format(model_path, binary=False)
    idx2word = dict(enumerate(emo2vec.index2word))
    n = 100  # len(idx2word)
    print(n, 'word vectors.')

    # invert idx -> word mapping
    word2idx = {w: i for i, w in {i: idx2word[i] for i in np.arange(n)}.items()}
    # word2idx = {w: i for i, w in idx2word.items()}

    t = np.empty((n, n), dtype='float16')

    for w1, i in word2idx.items():
        for w2, j in word2idx.items():
            t[i, j] = to_cosine_dist(emo2vec.similarity(w1, w2)) ** 2

    t /= sigma ** 2
    t = np.exp(-t)
    t = normalize(t, axis=0, norm='l1', copy=False)
    t = normalize(t, axis=1, norm='l1', copy=False)

    return t, word2idx


def labelprop(model, lexicon, sigma):
    print('Build T...')
    T, word2idx_T = build_T(model, sigma)
    print('Build Y...')
    Y, L = build_Y(lexicon, word2idx_T)

    # unlabeled nodes indices
    U = np.setdiff1d(np.asarray(list(word2idx_T.values()), dtype='int32'), L)

    new_order = np.append(U, L)
    T[:, :] = T[list(new_order)][:, list(new_order)]
    Y[:] = Y[new_order]

    T_uu = T[:len(U), :len(U)]
    T_ul = T[:len(U), len(U):]
    Y_l = Y[len(U):]
    Y_u = Y[:len(U)]

    n, m = T_uu.shape
    I = np.eye(n, m)

    clip_to_range_0_1(Y)
    Y_u = inv(I - T_uu) @ T_ul @ Y_l
    clip_to_range_0_1(Y)

    i2i = {old: new for (new, old) in enumerate(new_order)}
    word2idx_T = {w: i2i[old] for w, old in word2idx_T.items()}

    return normalize(Y, axis=1, norm='l1'), word2idx_T, - np.sum(Y * np.log(Y))


# for sigma in np.arange(start=0.1, stop=0.2, step=0.2):

sigma = 0.2

y, w2i, h = labelprop('resources/emotion_specific/bilstm_300d.txt',
                      'resources/data/emolex.txt', sigma)

with open('y_lp_1sigma.txt', 'w', encoding='utf-8') as f:
    for w, i in w2i.items():
        # i =
        print(w, str(y[i]).replace('\n   ', '   ').replace('[', '').replace(']', '').replace('  ', ''), file=f)
