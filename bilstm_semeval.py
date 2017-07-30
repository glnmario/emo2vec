from keras import optimizers, initializers
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.regularizers import l2

from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import pearsonr, spearmanr

import numpy as np

np.random.seed(13)

corpus_path = 'resources/data/semeval/headlines.txt'
labels_path = 'resources/data/semeval/labels.txt'
pretrained_model = 'resources/emotion_specific/bilstm_300d.txt'
batch_size = 128
embed_dim = 300
epochs = 80
train_over_test = 0.80
n_classes = 6
k = 0.25

def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


# texts[i] has labels[i]
texts = []
# labels = []
labels = []

print('Process corpus...')
with open(corpus_path, 'r') as f:
    for line in f:
        texts.append(line.strip())

with open(labels_path, 'r') as f:
    for line in f:
        l = list(map(int, (line.split())))
        labels.append(l)
        # labels.append(normalize(l))
print('Found %s texts and %s labels.' % (len(texts), len(labels)))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input
max_seq_len = np.max([len(s) for s in sequences])

word_to_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
V = len(word_to_index)  # vocabulary size

data = pad_sequences(sequences, maxlen=max_seq_len)
labels = np.asarray(labels) / 100
# for label in labels:
#     label[label >= k] = 1
#     label[label < k] = 0
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_train_samples = int(train_over_test * data.shape[0])

x_train = data[:num_train_samples]
x_test = data[num_train_samples:]
y_train = labels[:num_train_samples]
y_test = labels[num_train_samples:]

print('Index word vectors...')
embeddings_index = {}
with open(pretrained_model, 'r') as f:
    next(f)  # skip header
    for line in f:
        values = line.split()
        if len(values) != embed_dim + 1:  # probably an error occurred during tokenization
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))


lexicon = dict()
with open('resources/data/emolex.txt', 'r') as f:
    emo_idx = 0  # anger: 0, disgust: 1, fear: 2, joy: 3, sadness: 4, surprise: 5
    for l in f:
        lemma, emotion, has_emotion = read_emo_lemma(l)
        if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
            lexicon[lemma] = np.empty(shape=(n_classes,), dtype='float32')
        if emotion in ['positive', 'negative', 'anticipation', 'trust']:
            continue
        lexicon[lemma][emo_idx] = has_emotion
        if emo_idx < n_classes - 1:
            emo_idx += 1
        else:
            # reset index - next line contains a new lemma
            emo_idx = 0

# exp_lexicon = {}
# with open('y_300_3000-100-5.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         split = line.split()
#         if len(split) != n_classes + 1:  # probably an error occurred during tokenization
#             print(line)
#             continue
#         word = split[0]
#         probs = np.asarray(split[1:], dtype='float32')
#         exp_lexicon[word] = probs

for word, coef in embeddings_index.items():
    try:
        lexicon[word][lexicon[word] == 0] = 1e-10
        embeddings_index[word] = np.append(coef, lexicon[word])
    except KeyError:
        embeddings_index[word] = np.append(coef, np.random.uniform(size=n_classes))

print('Prepare embedding matrix...')
embedding_matrix = np.zeros((V + 1, embed_dim+n_classes))
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i - 1] = embedding_vector

print('Build model...')

model = Sequential()
model.add(Embedding(V + 1,
                    embed_dim+n_classes,
                    weights=[embedding_matrix],
                    input_length=max_seq_len,
                    trainable=True))
model.add(BatchNormalization())
model.add(Bidirectional(
    (LSTM(128, dropout=0.2, recurrent_dropout=0.4, recurrent_activation='sigmoid', recurrent_regularizer=l2()))))
model.add(BatchNormalization())
model.add(Dense(n_classes, activation='sigmoid'))

adagrad = optimizers.Adagrad(lr=0.002, epsilon=1e-08, decay=1e-04)

model.compile(loss='binary_crossentropy',
              optimizer=adagrad)

print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          validation_data=(x_test, y_test))

preds = model.predict(x_test, verbose=True)
print(preds.shape)
print(y_test.shape)

rs = []
rhos = []
for i in range(0, len(preds)):
    r, _ = pearsonr(y_test[i], preds[i])
    rho, _ = spearmanr(y_test[i], preds[i])
    rs.append(r)
    rhos.append(rho)

print('\nAverage Pearson r:', np.mean(rs))
print('Average Spearman rho:', np.mean(rhos))

for pred in preds:
    pred[pred >= k] = 1
    pred[pred < k] = 0

for coarse_y in y_test:
    coarse_y[coarse_y >= k] = 1
    coarse_y[coarse_y < k] = 0

print(classification_report(y_test, preds))
print("P: {}  R: {}  F1: {}".format(precision_score(y_test, preds, average='micro'),
                                    recall_score(y_test, preds, average='micro'),
                                    f1_score(y_test, preds, average='micro'))

