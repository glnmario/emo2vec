import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize
from math import floor, pi
import sys
np.random.seed(13)

if len(sys.argv) == 2:
    STOP = int(sys.argv[1])
elif len(sys.argv) == 1:
    STOP = 50000  # i.e. no number of words limit
else:
    sys.exit('Usage: lp_300.py [word limit]')

# # Lazy way to avoid overwriting old output files
# f = open('log_sigmas2.txt', 'x')
# f.close()

class Model:
    def __init__(self, n_labeled, n_unlabeled, input_dims, n_classes):
        self._t_uu = tuu = tf.placeholder(tf.float32, shape=[n_unlabeled, n_unlabeled])
        self._t_ul = tul = tf.placeholder(tf.float32, shape=[n_unlabeled, n_labeled])
        self._y_l = y_l = tf.placeholder(tf.float32, shape=[n_labeled, n_classes])

        tuu = tf.sigmoid(t_uu)
        tul = tf.sigmoid(t_ul)

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
        if line_idx >= STOP:
            break

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

w = np.asarray([2.75008321, 1.49768388, -0.4365142, -0.26589856, 2.64165592, 2.16741538, 2.01852489, 0.30227602, 1.11528945, 3.15220809, 1.36675048, 0.20136487, 1.19920945, 2.0775516, 1.47605801, -0.63313329, -0.17971444, 1.40345514, 0.9603231 , 1.85236061,  1.68006063,-0.51950377, 0.85443658, 1.78085256, -0.34736976, 1.68618941, 1.38618171, 1.4973731 , 3.2328639,  0.57973289, 1.99998391, 2.1845386 , -0.16269283,-0.95068461, 0.82642514, 2.67797565, 1.68234754, 2.11829543, 2.29444575, 1.75631928, 1.10478044, -1.0361582, 1.02232039,-0.20288679,  1.58657908, 1.42386854, 0.56387925, 0.14466508,  2.32827353, -0.59595078, 2.14662385, 0.54820102, 2.50744128, 2.37333894, 1.22021997,-0.44365704,  3.44509077, 2.6657052 ,-0.60511374,3.27951336, -0.38686621, 0.62073493, 0.84138227, 2.40600562, 0.01257626, -0.25159845, 2.31974316,-0.6953975 ,  0.75083864,-0.22232461,-0.24304244, 1.78589892,  0.87874699,-0.38370898, -0.44974527, 2.23025012, 2.83035874,  2.45414114, 0.25735155, 2.39575171, -0.37287751, 2.77704954, 2.49573088, 2.52037311,  1.43923092, 2.57060838, 3.05275798, 1.75145936, 1.97189212, 1.56806004, 1.11594629, 0.45332426,  0.44732654, 1.8817625 ,-0.62545353, 0.04992427, -0.72837752, 1.89396894, 1.66417003, 1.67852342, 3.05141163, 0.90650702, 2.07665348, 2.23708868,  0.14255558, 1.18810809,-0.15095305, 1.78856552,  2.13306093, 3.22739077, 2.26588631,2.55851579, 2.43340659, 2.72310543, 0.97403413, 1.57915223, -0.47983414,-0.58620292, 3.22506642, 1.88781023,  2.92870331,-0.16575032, 1.66830754, 3.01955843, 2.56348443, 0.37401107, -0.41656739, 2.38839507,  0.85059786, 1.58633256, 0.77152014, 1.37762487,  0.7851193, 2.45103383, -0.20724612, 2.72348404,-0.34559572, 1.22299004, -0.17044714,-0.52851152,  0.42245197, 0.49725473, 0.52012742, -0.19306304,  1.78320003, 3.01532483, 3.08560324,-1.04622626, 2.76314712, 0.66018778, 1.41386557, 0.28505296, -0.37574157, 1.07624722, 3.09865355, -0.61163813,  2.3936646 , 2.39342046, 0.84490764, 3.09656668, 2.52954364, 1.40159786, -0.7661429 , 0.44996163, -0.24063994, 2.06646466,-0.03058309, 1.09954321,  0.08101629, 2.66719437, 0.77760905, 1.96614146, 1.16308236, 2.29601932, 2.35631156, 2.86076403, -0.01725313, 2.33935142, 0.91386306, 0.39824608, 2.31827235, -0.89884406, -0.12857686, 1.72721291, 0.46043068, 0.52003396,-0.04424967,-0.67732751, -0.60715854,  2.60270643,-0.19803087, 1.08014441, 0.63962293, 1.25897312,  2.66742563,  0.73746771,-0.65795475, 0.6134755, -0.12001736, 1.26283741,  1.35586107,  2.59204793, 1.88603592, 2.82732439, 3.0847044 , 2.68180275, -0.25754142,  1.38219321,-0.28847447, 2.80754828, -0.55557495, 0.08034883,  0.78478163, -0.62458843, 1.79421115, 0.85786289, 2.45589399, 0.59645414,  2.58127785,  0.76136935, 1.31891608, 2.60387683, 1.96153927,-0.63796073,  2.73470497,  2.38942909, 2.19189239,-0.12120765, 3.4012239 ,-0.20221926,  2.80929995,  1.69832134,-0.64210528, 1.41273606, 2.51677728,-0.26460075,  2.05723262,  1.20340395, 1.11551476, 2.17704487, 2.08557248, 2.10255909,  0.00682311,  2.41622853, 0.884556, 2.00329638, 2.50358176, 0.47783926,  1.04998505,  0.91964149, 0.83547163, 0.10405838, 2.48301053, 1.67398083,  3.2068224 ,  2.17157149, 2.17904878, 0.81180036, 1.8699553 , 0.39552787,  3.06220055,  2.8039546 , 2.33392262, 2.94794917, 0.6445511 , 1.41906047,  2.47305655, -0.00730798, 2.97012424, 2.81670427, 1.36050904,-0.74320889,  2.23647046,  1.30356014, 0.32140392, 0.45754161, 0.19514999, 1.08816159,  1.58558345, -0.56351531, 2.19076753, 1.25380576, 2.32418251, 2.2023952 , -0.04442248,  2.14609504, 0.82815224, 2.34737897, 1.30651975, 0.29504114, -0.38945064,  1.62722659, 1.36140347,-0.12986997, 1.1746428 , 3.46422362, -0.19458437,  1.51364017, 2.26226211, 1.12018299])

b = 2.68437
print('Build distance matrix.')
t = np.empty((n, n), dtype='float32')

log_count = 0
for j in word2idx.values():
    for k in word2idx.values():
        t[j, k] = (w @ (embeddings[j] * embeddings[k])) + b
    log_count += 1
    if log_count % 1000 == 0:
        print(log_count, "/", n, sep="")


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

labeled_indices.sort()
l = labeled_indices
u = np.setdiff1d(np.asarray(list(word2idx.values()), dtype='int32'), l)

new_order = np.append(u, l)
t[:, :] = t[new_order][:, new_order]
y[:] = y[new_order]

T_uu = t[:len(u), :len(u)]
T_ul = t[:len(u), len(u):]
Y_l = y[len(u):]

i2i = np.zeros_like(new_order, dtype='int32')
for new, old in enumerate(new_order):
    i2i[old] = new

word2idx = {w: i2i[old] for w, old in word2idx.items()}

print('Tensorflow.')
sess = tf.Session()

with tf.variable_scope("model", reuse=False):
    model = Model(len(l), len(u), embed_dim, n_emotions)

sess.run(tf.global_variables_initializer())

Y = sess.run([model.y],
             {model.t_uu: T_uu, model.t_ul: T_ul, model.y_l: Y_l})

with open('y300_4000-1500-3.txt', 'w', encoding='utf-8') as f:
    for w, i in word2idx.items():
        print(w, str(y[i]).replace('\n   ', '   ').replace('[', '').replace(']', '').replace('  ', ' '), file=f)

sess.close()
