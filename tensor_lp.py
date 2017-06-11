import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.preprocessing import normalize
from math import floor, pi
import sys
np.random.seed(13)

if len(sys.argv) != 4:
    sys.exit('3 args required: batch size, # batches, # epochs')

class Model:
    def __init__(self, n_labeled, n_unlabeled, input_dims, n_classes):
        self._t_uu = t_uu = tf.placeholder(tf.float32, shape=[n_unlabeled, n_unlabeled, input_dims])
        self._t_ul = t_ul = tf.placeholder(tf.float32, shape=[n_unlabeled, n_labeled, input_dims])
        self._y_l = y_l = tf.placeholder(tf.float32, shape=[n_labeled, n_classes])

        sigmas_init = tf.truncated_normal(shape=[input_dims,], mean=0.5, stddev=0.2, dtype=tf.float32)
        # sigmas_init = tf.constant_initializer(1, dtype=tf.float32)
        self._sigmas = sigmas = tf.get_variable("sigmas", dtype=tf.float32, initializer=sigmas_init)
        sigmas = tf.Print(sigmas, [sigmas], 'Sigmas: ', summarize=30)

        tuu = tf.exp(- tf.reduce_sum(t_uu / (sigmas ** 2), axis=2))
        tul = tf.exp(- tf.reduce_sum(t_ul / (sigmas ** 2), axis=2))

        uniform_init = tf.constant_initializer(1 / n_unlabeled+n_labeled, dtype=tf.float32)
        u1 = tf.get_variable("uniform1",
                             dtype=tf.float32,
                             shape=[n_unlabeled, n_unlabeled],
                             trainable=False,
                             initializer=uniform_init)
        u2 = tf.get_variable("uniform2",
                             dtype=tf.float32,
                             shape=[n_unlabeled, n_labeled],
                             initializer=uniform_init)

        self._epsilon = epsilon = tf.get_variable("epsilon",
                                                  dtype=tf.float32,
                                                  shape=[],
                                                  trainable=True,
                                                  initializer=tf.constant_initializer(0.5e-4, dtype=tf.float32))
        epsilon = tf.Print(epsilon, [epsilon], 'Epsilon: ')
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
        inv = tf.matrix_solve_ls((I - tuu), I, l2_regularizer=1)

        y_u = tf.matmul(tf.matmul(inv, tul), y_l)

        self._y = y = tf.concat([y_u, y_l], 0)
        y = tf.clip_by_value(y, 1e-15, float("inf"))

        self._entropy = entropy = - tf.reduce_sum(y * tf.log(y))
        self._train_op = tf.train.AdamOptimizer(0.01).minimize(entropy)


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
    def train_op(self):
        return self._train_op

    @property
    def entropy(self):
        return self._entropy

    @property
    def sigmas(self):
        return self._sigmas

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

_embeddings = []
word2idx = {}
line_idx = 0
with open('resources/emotion_specific/bilstm_300d.txt', 'r', encoding='UTF-8') as f:
    next(f)  # skip header
    for line in f:
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



print('Build distance matrix.')
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

tot_batch_size = int(sys.argv[1])
n_batches = int(sys.argv[2])
epochs = int(sys.argv[3])
l_batch_size = int(tot_batch_size * (5869 / n))
u_batch_size = tot_batch_size - l_batch_size

print('Tensorflow.')
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config=config)

with tf.variable_scope("model", reuse=None) as scope:
    model = Model(l_batch_size, u_batch_size, NDIMS, NUM_EMOTIONS)

sess.run(tf.global_variables_initializer())

for batch in range(n_batches):
    rand_u = enumerate(np.random.choice(u, size=u_batch_size, replace=False))
    rand_l = enumerate(np.random.choice(l, size=l_batch_size, replace=False))

    yl_batch = np.asarray(y[ [j for (i, j) in rand_l] ])
    tuu_batch = np.empty((u_batch_size, u_batch_size, NDIMS), dtype='float16')
    tul_batch = np.empty((u_batch_size, l_batch_size, NDIMS), dtype='float16')

    for (j, j_) in rand_u:
        for (k, k_) in rand_u:
            tuu_batch[j, k] = (embeddings[j_] * embeddings[k_]) ** 2
        for (k, k_) in rand_l:
            tul_batch[j, k] = (embeddings[j_] * embeddings[k_]) ** 2

    for epoch in range(epochs):
        h, _, Y, sigmas, epsilon = sess.run([model.entropy, model.train_op, model.y, model.sigmas, model.epsilon],
                                                 {model.t_uu: tuu_batch, model.t_ul: tul_batch, model.y_l: yl_batch})

        print(h)
        if batch == n_batches - 1 and epoch == epochs - 1:
            with open('log_sigmas.txt', 'w') as f:
                print('Entropy:', h, '\n', file=f)
                print('Epsilon:', epsilon, '\n', file=f)
                print('Sigmas:', file=f)
                print([sigma for sigma in sigmas], file=f)

            np.savetxt('y_sigmas.txt', normalize(Y, axis=1, norm='l1', copy=False))

sess.close()
