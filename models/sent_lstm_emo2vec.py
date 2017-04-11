from __future__ import print_function
import csv
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
np.random.seed(13)

RESOURCES_PATH = 'resources/'
CORPUS_PATH = RESOURCES_PATH + 'twitter_sent_corpus_25k.csv'
OUTPUT_MODEL = RESOURCES_PATH + 'lstm_sent_300d.txt'
PRETRAINED_MODEL = RESOURCES_PATH + 'SGNS-300d.txt'
BATCH_SIZE = 32
EMBEDDING_DIM = 300
EPOCHS = 1
TRAIN_OVER_TEST = 0.7

# texts[i] has labels[i]
texts = []
labels = []

print('Process corpus...')
with open(CORPUS_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        texts.append(row[3].strip())
        labels.append(int(row[1]))
print('Found %s texts.' % len(texts))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input
max_seq_len = int(3 * np.median([len(s) for s in sequences]))

word_to_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
V = len(word_to_index)  # vocabulary size

data = pad_sequences(sequences, maxlen=max_seq_len)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_train_samples = int(TRAIN_OVER_TEST * data.shape[0])

x_train = data[:num_train_samples]
y_train = labels[:num_train_samples]
x_test = data[num_train_samples:]
y_test = labels[num_train_samples:]

print('Index word vectors...')
embeddings_index = {}
with open(PRETRAINED_MODEL, 'r') as f:
    next(f)  # skip header
    for line in f:
        values = line.split()
        if len(values) != EMBEDDING_DIM+1:  # probably an error occurred during tokenization
            continue
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

print('Prepare embedding matrix...')
embedding_matrix = np.zeros((V+1, EMBEDDING_DIM))
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i-1] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = True so as to let the embeddings vary
embedding_layer = Embedding(V+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            trainable=True)

print('Build model...')
sequence_input = Input(shape=(max_seq_len,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
preds = Dense(1, activation='sigmoid')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)

print('\nTest score:', score)
print('Test accuracy:', acc)

print('Write word vectors to', OUTPUT_MODEL)
with open(OUTPUT_MODEL, 'w') as f:
    f.write(" ".join([str(V), str(EMBEDDING_DIM)]))
    f.write("\n")

    vectors = model.get_weights()[0]  # shape: V x EMBEDDING_DIM

    for word, i in word_to_index.items():
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vectors[i, :]))))
        f.write("\n")
