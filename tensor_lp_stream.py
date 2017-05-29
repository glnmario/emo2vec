import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.preprocessing import normalize

np.random.seed(13)


def stream(n_unlabeled, part_unlabeled, part_labeled, input_dims):
    tuu_part_in = tf.placeholder(tf.float32, shape=[n_unlabeled, part_unlabeled, input_dims])
    tul_part_in = tf.placeholder(tf.float32, shape=[n_unlabeled, part_labeled, input_dims])


    sigmas_init = tf.truncated_normal(shape=[input_dims,], mean=1.0, stddev=0.45, dtype=tf.float32)

    try:
        sigmas = tf.get_variable("sigmas", dtype=tf.float32, initializer=sigmas_init)
    except ValueError:
        scope.reuse_variables()
        sigmas = tf.get_variable("sigmas")

    tuu_part_out = tf.exp(- tf.reduce_sum(tuu_part_in / (sigmas ** 2), axis=2))
    tul_part_out = tf.exp(- tf.reduce_sum(tul_part_in / (sigmas ** 2), axis=2))

    return tuu_part_in, tul_part_in, tuu_part_out, tul_part_out


class Model:
    def __init__(self, tuu, tul, n_labeled, n_unlabeled, n_classes):

        self._y_l = y_l = tf.placeholder(tf.float32, shape=[n_labeled, n_classes])

        uniform_init = tf.constant_initializer(1 / n_unlabeled+n_labeled, dtype=tf.float32)
        u1 = tf.get_variable("uniform1",
                             dtype=tf.float32,
                             shape=[n_unlabeled, n_unlabeled],
                             trainable=False,
                             initializer=uniform_init)
        u2 = tf.get_variable("uniform2",
                             dtype=tf.float32,
                             shape=[n_unlabeled, n_labeled],
                             trainable=False,
                             initializer=uniform_init)

        self._epsilon = epsilon = tf.get_variable("epsilon",
                                                  dtype=tf.float32,
                                                  shape=[],
                                                  initializer=tf.constant_initializer(0.5e-4))
        tuu = epsilon * u1 + (1 - epsilon) * tuu
        tul = epsilon * u2 + (1 - epsilon) * tul

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

        # self._lagr = lagr = - 100 * tf.minimum(0., tf.reduce_min(y_u))
        y = tf.concat([y_u, y_l], 0)
        self._y = y = tf.clip_by_value(y, 1e-15, float("inf"))

        self._entropy = entropy = - tf.reduce_sum(y * tf.log(y))
        self._train_op = tf.train.AdamOptimizer(0.1).minimize(entropy)

    @property
    def y_l(self):
        return self._y_l

    @property
    def y(self):
        return self._y

    @property
    def train_op(self):
        return self._train_op

    @property
    def entropy(self):
        return self._entropy

    @property
    def epsilon(self):
        return self._epsilon


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])



NUM_EMOTIONS = 6
NDIMS = 300
STOP = 40000

_embeddings = []
word2idx = {}
line_idx = 0
with open('resources/emotion_specific/bilstm_300d.txt', 'r', encoding='UTF-8') as f:
    next(f)  # skip header
    for line in f:
        if line_idx >= STOP:
            break
        values = line.split()
        if len(values) != NDIMS + 1:
            # probably an error occurred durtokenization
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float16')
        _embeddings.append(coefs)
        word2idx[word] = line_idx
        line_idx += 1
print('Found', line_idx-1, 'word vectors.')
n = line_idx
embeddings = np.asarray(_embeddings, dtype='float16')

y_l = np.empty(shape=(14182, NUM_EMOTIONS), dtype='float16')
lexeme2index = dict()
with open('resources/data/emolex.txt', 'r') as f:
    emo_idx = 0  # anger: 0, disgust: 1, fear: 2, joy: 3, sadness: 4, surprise: 6
    i = 0
    for l in f:
        lemma, emotion, has_emotion = read_emo_lemma(l)
        if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
            lexeme2index[lemma] = i
        if emotion in ['positive', 'negative', 'anticipation', 'trust']:
            continue
        y_l[i][emo_idx] = has_emotion
        if emo_idx < NUM_EMOTIONS - 1:
            emo_idx += 1
        else:
            # reset index - next line contains a new lemma
            emo_idx = 0
            i += 1

print('Initialize label distribution matrix.')
y = np.random.random((n, NUM_EMOTIONS))

labeled_indices = []
for word, idx in lexeme2index.items():
    try:
        # if word in corpus
        idx_T = word2idx[word]  # get index of word in T
        y[idx_T] = y_l[idx]  # set values of labeled word
        labeled_indices.append(idx_T)
    except KeyError:
        continue

# turn multi-labels into prob distribution
y = normalize(y, axis=1, norm='l1', copy=False)

l = labeled_indices
u = np.setdiff1d(np.asarray(list(word2idx.values()), dtype='int32'), l)
Y_l = y[l]

print('Build distance matrix.')
part_size = 200
n_iter= n // part_size
last_part = n - (n_iter)*part_size

sess = tf.Session()

for it in range(n_iter):
    # tuu_part = None
    # tul_part = None

    if it == n_iter - 1:
        size = last_part
    else:
        size = part_size

    print(it, size)

    _t = np.empty((n, size, NDIMS), dtype='float16')
    for j in sorted(word2idx.values()):
        for k in sorted(word2idx.values()):

            if k >= it * part_size and k < (it+1) * size:
                _t[j % part_size, k % part_size] = (embeddings[j] - embeddings[k]) ** 2

    _u = np.asarray([idx for idx in u if idx >= it * part_size and idx < (it+1) * size])
    _l = np.asarray([idx for idx in l if idx >= it * part_size and idx < (it+1) * size])

    T_uu = _t[u][:, _u % part_size]
    T_ul = _t[u][:, _l % part_size]

    with tf.variable_scope("model", reuse=None) as scope:
        _tuu_part_in, _tul_part_in, _tuu_part_out, _tul_part_out = stream(len(u), len(_u), len(_l), NDIMS)

    sess.run(tf.global_variables_initializer())
    _tuu_part, _tul_part = sess.run([_tuu_part_out, _tul_part_out],
                                    {_tuu_part_in: T_uu, _tul_part_in: T_ul})
    if it == 0:
        tuu_part = _tuu_part
        tul_part = _tul_part
    else:
        tuu_part = tf.concat([tuu_part, _tuu_part], axis=1)
        tul_part = tf.concat([tul_part, _tul_part], axis=1)

Tuu = tuu_part
Tul = tul_part


model = Model(Tuu, Tul, len(l), len(u), NUM_EMOTIONS)
sess.run(tf.global_variables_initializer())

rng = 50
for i in range(rng):
    h, _, Y, eps = sess.run([model.entropy, model.train_op, model.y, model.epsilon], {model.y_l: Y_l})
    print(h, sep='\n\n')

    if i == rng-1:
        print(normalize(Y, norm='l1', axis=1, copy=False))

sess.close()
