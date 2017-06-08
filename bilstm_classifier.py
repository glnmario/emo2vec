from __future__ import print_function
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np
from keras import optimizers
from sklearn.metrics import classification_report
import hashtag_corpus
np.random.seed(13)

PRETRAINED_MODEL = 'resources/emotion_specific/bilstm_300d.txt'
LEXICON_PATH = 'resources/data/emolex.txt'
BATCH_SIZE = 64
EMBEDDING_DIM = 300
EPOCHS = 30
TRAIN_SIZE = 0.7
VALID_SIZE = 0.0
NUM_EMOTIONS = 6


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


x_train, y_train, x_val, y_val, x_test, y_test = hashtag_corpus.split(TRAIN_SIZE, VALID_SIZE)
word_to_index = hashtag_corpus.word_index()
V = len(word_to_index)

print('Index word vectors...')
embeddings_index = {}
with open(PRETRAINED_MODEL, 'r', encoding='UTF-8') as f:
    next(f)  # skip header
    for line in f:
        values = line.split()
        if len(values) != EMBEDDING_DIM + 1:  # probably an error occurred during tokenization
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

lexicon = dict()

with open(LEXICON_PATH, 'r') as f:
    emo_idx = 0  # anger: 0, disgust: 1, fear: 2, joy: 3, sadness: 4, surprise: 5
    for l in f:
        lemma, emotion, has_emotion = read_emo_lemma(l)
        if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
            lexicon[lemma] = np.empty(shape=(NUM_EMOTIONS,))
        if emotion in ['positive', 'negative', 'anticipation', 'trust']:
            continue
        lexicon[lemma][emo_idx] = has_emotion
        if emo_idx < NUM_EMOTIONS - 1:
            emo_idx += 1
        else:
            # reset index - next line contains a new lemma
            emo_idx = 0

for word, coef in embeddings_index.items():
    try:
        embeddings_index[word] = np.append(coef, lexicon[word])
    except KeyError:
        embeddings_index[word] = np.append(coef, np.random.uniform(size=NUM_EMOTIONS))


print('Prepare embedding matrix...')
embedding_matrix = np.zeros((V + 1, EMBEDDING_DIM+NUM_EMOTIONS))
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i - 1] = embedding_vector

print('Build model...')

model = Sequential()
model.add(Embedding(V + 1,
                    EMBEDDING_DIM+NUM_EMOTIONS,
                    weights=[embedding_matrix],
                    input_length=hashtag_corpus.max_sequence_len(),
                    trainable=True))
model.add(BatchNormalization())
model.add(Bidirectional(
    (LSTM(128, dropout=0.1, recurrent_dropout=0.2, recurrent_activation='sigmoid', recurrent_regularizer=l2()))))
model.add(BatchNormalization())
model.add(Dense(NUM_EMOTIONS, activation='softmax'))

adagrad = optimizers.Adagrad(lr=0.005, epsilon=1e-08, decay=1e-5)

model.compile(loss='categorical_crossentropy',
              optimizer=adagrad)

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    shuffle=True,
                    validation_data=(x_val, y_val))

print(history.history)

preds = model.predict_classes(x_test, verbose=True)
Y = [np.argmax(x) for x in y_test]

print(classification_report(Y, preds))


# # serialize model to JSON
# model_name = "saved_models/bilstm_btc128_ep20_dr1-2_lr005_dec5_300d_catcross_nobatch_nol2"
# model_json = model.to_json()
# with open(model_name + ".json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights(model_name + ".h5")
# print("Saved model to disk")
