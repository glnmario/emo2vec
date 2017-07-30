from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.regularizers import l2
from keras import optimizers

from sklearn.metrics import precision_score, recall_score, f1_score

import hashtag_corpus

import numpy as np

np.random.seed(13)

pretrained_model = 'resources/emotion_specific/bilstm_300d.txt'
lexicon_path = 'resources/data/emolex.txt'
batch_size = 64
embed_dim = 300
epochs = 70
train_size = 0.6
valid_size = 0.2
n_classes = 6


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


x_train, y_train, x_val, y_val, x_test, y_test = hashtag_corpus.split(train_size, valid_size)
word_to_index = hashtag_corpus.word_index()
V = len(word_to_index)

print('Index word vectors...')
embeddings_index = {}
with open(pretrained_model, 'r', encoding='UTF-8') as f:
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

with open(lexicon_path, 'r') as f:
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
                    input_length=hashtag_corpus.max_sequence_len(),
                    trainable=True))

model.add(BatchNormalization())
model.add(Bidirectional(LSTM(128,
                             dropout=0.1,
                             recurrent_dropout=0.2,
                             recurrent_activation='sigmoid',
                             recurrent_regularizer=l2())))

model.add(BatchNormalization())
model.add(Dense(n_classes, activation='sigmoid'))

adagrad = optimizers.Adagrad(lr=0.005, epsilon=1e-08, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=adagrad)

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    validation_data=(x_val, y_val))

print(history.history)

preds = model.predict_classes(x_test, verbose=True)
Y = [np.argmax(x) for x in y_test]

print("P: {}  R: {}  F1: {}".format(precision_score(Y, preds, average='micro'),
                                    recall_score(Y, preds, average='micro'),
                                    f1_score(Y, preds, average='micro')))
