import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import normalize
from pandas import DataFrame
from math import acos, pi
from scipy.stats import entropy
import sys
np.random.seed(13)

if len(sys.argv) == 2:
    STOP = int(sys.argv[1])
elif len(sys.argv) == 1:
    STOP = 50000  # i.e. no number of words limit
else:
    sys.exit('Usage: crossval_1.py [word limit]')


def clip_to_range_0_1(v):
    v[v <= 0.] = 1e-15
    v[v > 1.] = 1.0


def kl_divergence(true, pred):
    clip_to_range_0_1(true)
    clip_to_range_0_1(pred)
    return entropy(true, pred)


def partition (list, n):
    return [list[i::n] for i in range(n)]


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


class Model:
    def __init__(self, n_labeled, n_unlabeled, n_classes):
        self._t_uu = t_uu = tf.placeholder(tf.float32, shape=[n_unlabeled, n_unlabeled])
        self._t_ul = t_ul = tf.placeholder(tf.float32, shape=[n_unlabeled, n_labeled])
        self._y_l = y_l = tf.placeholder(tf.float32, shape=[n_labeled, n_classes])

        self._w = w = tf.placeholder(tf.float32, shape=[])
        self._b = b = tf.placeholder(tf.float32, shape=[])

        tuu = tf.sigmoid(w * t_uu + b)
        tul = tf.sigmoid(w * t_ul + b)

        # column normalization
        tuu_col_norms = tf.norm(tuu, ord=1, axis=0)
        tul_col_norms = tf.norm(tul, ord=1, axis=0)
        tuu /= tuu_col_norms
        tul /= tul_col_norms

        # row normalization
        tuu_row_norms = tf.norm(tuu, ord=1, axis=1)
        tul_row_norms = tf.norm(tul, ord=1, axis=1)
        tuu /= tf.reshape(tuu_row_norms, [n_unlabeled, 1])
        tul /= tf.reshape(tul_row_norms, [n_unlabeled, 1])

        I = tf.eye(n_unlabeled, dtype=tf.float32)
        inv = tf.matrix_solve_ls((I - tuu), I, l2_regularizer=0.01)

        y_u = tf.matmul(tf.matmul(inv, tul), y_l)

        y = tf.concat([y_u, y_l], 0)
        self._y = y = tf.clip_by_value(y, 1e-15, float("inf"))


    @property
    def t_uu(self):
        return self._t_uu

    @property
    def t_ul(self):
        return self._t_ul

    @property
    def y_l(self):
        return self._y_l

    @property
    def y(self):
        return self._y

    @property
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


NUM_EMOTIONS = 6
NDIMS = 300

_embeddings = []
word2idx = {}
line_idx = 0
with open('resources/emotion_specific/bilstm_300d.txt', 'r', encoding='UTF-8') as f:
    next(f)  # skip header

    for line in f:
        if line_idx >= STOP:
            break

        values = line.split()

        # probably an error occurred during tokenization
        if len(values) != NDIMS + 1:
            continue

        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')

        # skip all-zeros vectors
        if not coefs.any():
            continue

        # only one vector for each word
        try:
            word2idx[word]
        except:
            _embeddings.append(coefs)
            word2idx[word] = line_idx
            line_idx += 1

n = line_idx
print('Found', n, 'word vectors.')

embeddings = np.asarray(_embeddings, dtype='float32')
embeddings = normalize(embeddings, axis=1, norm='l2', copy=False)

print('Build distance matrix.')
t = np.empty((n, n), dtype='float32')

log_count = 0
for j in word2idx.values():
    for k in word2idx.values():
        t[j, k] = embeddings[j] @ embeddings[k]

    log_count += 1
    if log_count % 1000 == 0:
        print(log_count, "/", n, sep="")

_lexicon = dict()
lexicon = dict()
lemma2index = dict()

with open('resources/data/emolex.txt', 'r') as f:
    emo_idx = 0  # anger: 0, disgust: 1, fear: 2, joy: 3, sadness: 4, surprise: 6
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

intersection = [idx for lemma, idx in lemma2index.items() if lemma in word2idx.keys()]
indices = list(intersection)
random.shuffle(indices)

partitions = partition(indices, 10)

print('Tensorflow.')
sess = tf.Session()

kls = []
for p in partitions:
    lex_arr = np.asarray(DataFrame(lexicon, dtype='float32').T.fillna(0), dtype='float32')
    y = np.random.random((len(word2idx), NUM_EMOTIONS))

    labeled_indices = []
    held_out_indices = []
    i2i_heldout = {}
    y_test = np.empty((len(p), NUM_EMOTIONS))

    i = 0
    for word, idx in lemma2index.items():
        try:
            # if word in corpus
            idx_T = word2idx[word]  # get index of word in corpus

            if idx in p:
                held_out_indices.append(idx_T)
                y_test[i] = lex_arr[idx]
                i2i_heldout[idx_T] = i
                i += 1
            else:
                labeled_indices.append(idx_T)
                y[idx_T] = lex_arr[idx]  # set values of labeled word
        except KeyError:
            continue

    y = normalize(y, axis=1, norm='l1', copy=False)

    labeled_indices.sort()
    l = labeled_indices
    u = np.setdiff1d(np.asarray(list(word2idx.values()), dtype='int32'), l)

    new_order = np.append(u, l)
    t[:, :] = t[new_order][:, new_order]
    y[:] = y[new_order]

    T_uu = t[:len(u), :len(u)]
    T_ul = t[:len(u), len(u):]
    Y_l = y[len(u):]

    old2new = np.zeros_like(new_order, dtype='int32')
    for new, old in enumerate(new_order):
        old2new[old] = new

    with tf.variable_scope("model", reuse=False):
        model = Model(len(l), len(u), NUM_EMOTIONS)

    sess.run(tf.global_variables_initializer())

    Y = sess.run([model.y],
                 {model.t_uu: T_uu, model.t_ul: T_ul, model.y_l: Y_l, model.w: 0.67309, model.b: 0.19266})

    Y = np.array(Y)
    Y = Y.reshape(Y.shape[1], Y.shape[2])

    divergences = []
    for idx in held_out_indices:
        i = i2i_heldout[idx]
        i_ = old2new[idx]
        divergences.append(kl_divergence(y_test[i], Y[i_]))

    kl = np.sum(np.asarray(divergences)) / len(divergences)
    kls.append(kl)
    print(kl)

result = np.sum(np.asarray(kls)) / len(kls)
print('Average kl:', result)

sess.close()
