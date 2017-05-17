from __future__ import print_function
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import KeyedVectors
from math import acos, pi

np.random.seed(13)

NUM_EMOTIONS = 8
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


def build_Y(lexicon_path, word2index_T):
    """
    :param lexicon_path: emotion lexicon
    :param word2index_T: dictionary word -> index referring to T matrix
    :return: (Y matrix, indices of labeled instances)
    """
    lexicon = dict()
    lexeme2index = dict()

    with open(lexicon_path, 'r') as f:
        emo_idx = 0  # anger: 0, anticipation: 1, disgust: 2, fear: 3, joy: 4, sadness: 5, surprise: 6, trust: 7
        i = 0
        for l in f:
            lemma, emotion, has_emotion = read_emo_lemma(l)
            if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
                lexicon[i] = np.empty(shape=(NUM_EMOTIONS,))
                lexeme2index[lemma] = i
            if emotion == 'positive' or emotion == 'negative':
                continue
            lexicon[i][emo_idx] = has_emotion
            if emo_idx < NUM_EMOTIONS - 1:
                emo_idx += 1
            else:
                # reset index - next line contains a new lemma
                emo_idx = 0
                i += 1

    y_l = np.asarray(DataFrame(lexicon, dtype='float16').T.fillna(0), dtype='float16')
    y = np.random.random((len(word2index_T), 8))

    labeled_indices = []
    for word, idx in lexeme2index.items():
        try:
            # if word in corpus
            idx_T = word2index_T[word]  # get index of word in T
            y[idx_T] = y_l[idx]  # set values of labeled word
            labeled_indices.append(idx_T)
        except KeyError:
            continue

    y = normalize(y, axis=1, norm='l1')  # turn multi-labels into prob distribution

    return y, labeled_indices


def build_T(model_path, sigma):
    """
    :param model_path: the path of the final model
    :return: (T matrix, word-to-index dictionary for T)
    """
    emo2vec = KeyedVectors.load_word2vec_format(model_path, binary=False)  # .init_sims(replace=True)
    idx2word = dict(enumerate(emo2vec.index2word))
    n = len(idx2word)

    # invert idx -> word mapping
    word2idx = {w: i for i, w in {i: idx2word[i] for i in np.arange(n)}.items()}

    t = np.empty((n, n), dtype='float16')

    for w1, i in word2idx.items():
        for w2, j in word2idx.items():
            t[i, j] = to_cosine_dist(emo2vec.similarity(w1, w2)) ** 2

    t /= sigma ** 2
    t = np.exp(-t)
    t = normalize(t, axis=0, norm='l1')
    t = normalize(t, axis=1, norm='l1')

    return t, word2idx


def labelprop(model, lexicon, sigma):
    print('Build T...')
    T, word2idx_T = build_T(model, sigma)
    print('Build Y...')
    Y, L = build_Y(lexicon, word2idx_T)

    U = np.setdiff1d(np.asarray(list(word2idx_T.values()), dtype='int32'), L)  # unlabeled nodes indices

    T_uu = T[U][:, U]
    T_ul = T[U][:, L]

    n, m = T_uu.shape
    I = np.eye(n, m)

    Y[U] = np.linalg.inv(I - T_uu) @ T_ul @ Y[L]
    # Y[U] = normalize(Y[U], axis=1, norm='l1')

    Y[Y == 0.] = 1e-15
    return - np.sum(Y * np.log(Y))


for sigma in np.arange(start=0.2, stop=2.2, step=0.2):
    print(sigma, labelprop('resources/emotion_specific/bilstm_300d.txt', 'resources/data/emolex.txt', sigma))


