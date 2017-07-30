import numpy as np
import tensorflow as tf
from pandas import DataFrame
from sklearn.preprocessing import normalize
from math import floor, pi
import sys
np.random.seed(13)

if len(sys.argv) != 4:
    sys.exit('Usage: batch_size, num_batches, num_epochs')

# # Lazy way to avoid overwriting old output files
# f = open('log_sigmas2.txt', 'x')
# f.close()

class Model:
    def __init__(self, n_labeled, n_unlabeled, n_classes):
        self._t_uu = t_uu = tf.placeholder(tf.float32, shape=[n_unlabeled, n_unlabeled])
        self._t_ul = t_ul = tf.placeholder(tf.float32, shape=[n_unlabeled, n_labeled])
        self._y_l = y_l = tf.placeholder(tf.float32, shape=[n_labeled, n_classes])

        w_init = tf.random_uniform(shape=[], minval=0.5, maxval=5)
        self._w = w = tf.get_variable("w", dtype=tf.float32, initializer=w_init)

        b_init = tf.random_uniform(shape=[], minval=-1, maxval=1)
        self._b = b = tf.get_variable("b", dtype=tf.float32, initializer=b_init)

        tuu = tf.sigmoid(w * t_uu + b)
        tul = tf.sigmoid(w * t_ul + b)
        # tuu = tf.Print(tuu, [tuu], 'tuu', summarize=30)
        # tul = tf.Print(tul, [tul], 'tul', summarize=30)

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

        self._entropy = entropy = - tf.reduce_sum(y * tf.log(y))
        self._train_op = tf.train.AdamOptimizer(0.005).minimize(entropy)


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
    def w(self):
        return self._w

    @property
    def b(self):
        return self._b


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


n_emotions = 6
embed_dim = 300

_embeddings = []
word2idx = {}
line_idx = 0
with open('resources/emotion_specific/bilstm_300d.txt', 'r', encoding='UTF-8') as f:
    next(f)  # skip header

    for line in f:
        values = line.split()

        # probably an error occurred during tokenization
        if len(values) != embed_dim + 1:
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
y_l = np.empty(shape=(14182, n_emotions), dtype='float32')

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
        if emo_idx < n_emotions - 1:
            emo_idx += 1
        else:
            # reset index - next line contains a new lemma
            emo_idx = 0
            i += 1

print('Initialize label distribution matrix.')
y = np.random.random((n, n_emotions))

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
n_epochs = int(sys.argv[3])
l_batch_size = int(tot_batch_size * (5869 / n))
u_batch_size = tot_batch_size - l_batch_size

with open('log_1sigma_btc_5000-1000-3.txt', 'w') as f:
    print('Batch size:', tot_batch_size, file=f)
    print('Number of batchs:', n_batches, file=f)
    print('Number of epochs:', n_epochs, "\n", file=f)

print('Tensorflow.')
sess = tf.Session()

with tf.variable_scope("model", reuse=None) as scope:
    model = Model(l_batch_size, u_batch_size, n_emotions)

sess.run(tf.global_variables_initializer())

f = open('log_1sigma_btc_5000-1000-3.txt', 'a')

for batch in range(n_batches):
    print('Batch', batch)
    rand_u = np.random.choice(u, size=u_batch_size, replace=False)
    rand_l = np.random.choice(l, size=l_batch_size, replace=False)

    yl_batch = np.asarray(y[rand_l])
    tuu_batch = np.empty((u_batch_size, u_batch_size), dtype='float32')
    tul_batch = np.empty((u_batch_size, l_batch_size), dtype='float32')

    for (j, j_) in enumerate(rand_u):
        for (k, k_) in enumerate(rand_u):
            tuu_batch[j, k] = embeddings[j_] @ embeddings[k_]
        for (k, k_) in enumerate(rand_l):
            tul_batch[j, k] = embeddings[j_] @ embeddings[k_]

    for epoch in range(n_epochs):
        h, _, w, b = sess.run([model.entropy, model.train_op, model.w, model.b],
                              {model.t_uu: tuu_batch, model.t_ul: tul_batch, model.y_l: yl_batch})

        print(epoch, ': ', h, sep='')

        if batch == n_batches - 1 and epoch == n_epochs - 1:
            print('\n\nEntropy:', h, file=f)
            print('w:', w, '\n', file=f)
            print('b', b, '\n', file=f)
            print('Output saved to emo2vec/log_1sigma_btc_5000-1000-3.txt')

f.close()
sess.close()
