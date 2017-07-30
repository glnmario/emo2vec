from keras import optimizers
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.regularizers import l2

from sklearn.metrics import precision_score, recall_score, f1_score

import hashtag_corpus

import numpy as np

np.random.seed(13)

output_model = 'resources/emotion_specific/bilstm_300d_sigmoid.txt'
pretrained_model = 'resources/pretrained/encow14ax-300-mincount-150-window-5-cbow.txt'
batch_size = 64
embed_dim = 300
epochs = 30
train_size = 0.6
valid_size = 0.2
n_classes = 6

x_train, y_train, x_val, y_val, x_test, y_test = hashtag_corpus.split(train_size, valid_size)
word_to_index = hashtag_corpus.word_index()
V = len(word_to_index)

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

print('Prepare embedding matrix...')
embedding_matrix = np.zeros((V + 1, embed_dim))
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i - 1] = embedding_vector

print('Build model...')

model = Sequential()
model.add(Embedding(V + 1,
                    embed_dim,
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

adagrad = optimizers.Adagrad(lr=0.005, epsilon=1e-08, decay=1e-4)
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

print('Write word vectors to', output_model)
with open(output_model, 'w') as f:
    f.write(" ".join([str(V), str(embed_dim)]))
    f.write("\n")

    vectors = model.get_weights()[0]  # shape: V x embed_dim

    for word, i in word_to_index.items():
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, vectors[i])))
        f.write("\n")
