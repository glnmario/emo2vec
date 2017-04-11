from __future__ import print_function

from gensim.models import KeyedVectors
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
np.random.seed(13)

RESOURCES_PATH = 'resources/'
CORPUS_PATH = RESOURCES_PATH + 'twitter_corpus.txt'
OUTPUT_MODEL = RESOURCES_PATH + 'antsyn_lstm_100d.txt'
PRETRAINED_MODEL = RESOURCES_PATH + 'wiki_en_dLCE_100d_minFreq_100.bin'  #  Wikipedia corpus, 100dim, min-count=100
BATCH_SIZE = 32
EMBEDDING_DIM = 100
EPOCHS = 2
TRAIN_OVER_TEST = 0.7

labels_index = {'anger': 0,
                'anticipation': 1,
                'disgust': 2,
                'fear': 3,
                'joy': 4,
                'sadness': 5,
                'surprise': 6,
                'trust': 7}
NUM_EMOTIONS = len(labels_index)

# texts[i] has emotion_labels[i]
texts = []
emotion_labels = []

print('Process corpus...')
with open(CORPUS_PATH, 'r') as f:
    for line in f:
        line_split = line[20:].split(sep='\t:: ')
        texts.append(line_split[0].strip())
        emotion_labels.append(labels_index[line_split[1].strip()])
print('Found %s texts.' % len(texts))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input
max_seq_len = np.max([len(s) for s in sequences])

word_to_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
V = len(word_to_index)  # vocabulary size

data = pad_sequences(sequences, maxlen=max_seq_len)
emotion_labels = to_categorical(np.asarray(emotion_labels), NUM_EMOTIONS)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', emotion_labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
emotion_labels = emotion_labels[indices]
num_train_samples = int(TRAIN_OVER_TEST * data.shape[0])

x_train = data[:num_train_samples]
y_train = emotion_labels[:num_train_samples]
x_test = data[num_train_samples:]
y_test = emotion_labels[num_train_samples:]

# Load pre-trained embeddings with lexical contrast information
w2v = KeyedVectors.load_word2vec_format(PRETRAINED_MODEL, binary=True)

print('Prepare embedding matrix...')
embedding_matrix = np.zeros((V+1, EMBEDDING_DIM))
for word, i in word_to_index.items():
    try:
        embedding_matrix[i-1] = w2v[word]
    except KeyError:
        continue  # words not found in embedding index will be all-zeros.

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

x = LSTM(128, dropout=0.19, recurrent_dropout=0.19)(embedded_sequences)
preds = Dense(NUM_EMOTIONS, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
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
