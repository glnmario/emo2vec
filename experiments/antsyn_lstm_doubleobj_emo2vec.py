from __future__ import print_function

import csv

from gensim.models import KeyedVectors
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
np.random.seed(13)

RESOURCES_PATH = 'resources/'
EMO_CORPUS_PATH = RESOURCES_PATH + 'twitter_corpus.txt'
SENT_CORPUS_PATH = RESOURCES_PATH + 'twitter_sent_corpus_25k.csv'
OUTPUT_MODEL = RESOURCES_PATH + 'antsyn_lstm_100d.txt'
PRETRAINED_MODEL = RESOURCES_PATH + 'wiki_en_dLCE_100d_minFreq_100.bin'  #  Wikipedia corpus, 100dim, min-count=100
BATCH_SIZE = 32
EMBEDDING_DIM = 100
EPOCHS = 1
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
emo_texts = []
emo_labels = []

print('Process corpus...')
with open(EMO_CORPUS_PATH, 'r') as f:
    for line in f:
        line_split = line[20:].split(sep='\t:: ')
        emo_texts.append(line_split[0].strip())
        emo_labels.append(labels_index[line_split[1].strip()])
print('Found %s texts.' % len(emo_texts))

emo_tokenizer = Tokenizer()
emo_tokenizer.fit_on_texts(emo_texts)
emo_sequences = emo_tokenizer.texts_to_sequences(emo_texts)  # one sequence of tokens per text input
max_seq_len = np.max([len(s) for s in emo_sequences])

emo_word_to_index = emo_tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
V1 = len(emo_word_to_index)  # vocabulary size

emo_data = pad_sequences(emo_sequences, maxlen=max_seq_len)
emo_labels = to_categorical(np.asarray(emo_labels), NUM_EMOTIONS)
print('Shape of data tensor:', emo_data.shape)
print('Shape of label tensor:', emo_labels.shape)

# split the data into a training set and a validation set
emo_indices = np.arange(emo_data.shape[0])
np.random.shuffle(emo_indices)
emo_data = emo_data[emo_indices]
emo_labels = emo_labels[emo_indices]
num_train_samples = int(TRAIN_OVER_TEST * emo_data.shape[0])

x1_train = emo_data[:num_train_samples]
y1_train = emo_labels[:num_train_samples]
x1_test = emo_data[num_train_samples:]
y1_test = emo_labels[num_train_samples:]


sent_texts = []
sent_labels = []

print('Process corpus 2...')
with open(SENT_CORPUS_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        sent_texts.append(row[3].strip())
        sent_labels.append(int(row[1]))
print('Found %s texts.' % len(sent_texts))

sent_tokenizer = Tokenizer()
sent_tokenizer.fit_on_texts(sent_texts)
sent_sequences = sent_tokenizer.texts_to_sequences(sent_texts)  # one sequence of tokens per text input

sent_word_to_index = sent_tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
V2 = len(sent_word_to_index)  # vocabulary size

sent_data = pad_sequences(sent_sequences, maxlen=max_seq_len)
sent_labels = np.asarray(sent_labels)
print('Shape of data tensor:', sent_data.shape)
print('Shape of label tensor:', sent_labels.shape)

# split the data into a training set and a validation set
sent_indices = np.arange(sent_data.shape[0])
np.random.shuffle(sent_indices)
sent_data = sent_data[sent_indices]
sent_labels = sent_labels[sent_indices]
num_train_samples = int(TRAIN_OVER_TEST * sent_data.shape[0])

x2_train = sent_data[:num_train_samples]
y2_train = sent_labels[:num_train_samples]
x2_test = sent_data[num_train_samples:]
y2_test = sent_labels[num_train_samples:]

# Load pre-trained embeddings with lexical contrast information
w2v = KeyedVectors.load_word2vec_format(PRETRAINED_MODEL, binary=True)

print('Prepare embedding matrix...')
word_to_index = b3 = [x for x in emo_word_to_index.items() if x in sent_word_to_index.items()]

V = None  # TODO: V1 + V2 - V1&V2
embedding_matrix = np.zeros((V+1, EMBEDDING_DIM))
j = 0
for word, i in emo_word_to_index.items():
    j = i
    try:
        embedding_matrix[i-1] = w2v[word]
    except KeyError:
        continue  # words not found in embedding index will be all-zeros.

for word, i in sent_word_to_index.items():
    try:
        embedding_matrix[j+i-1] = w2v[word]
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

x = LSTM(64, dropout=0.07, recurrent_dropout=0.17)(embedded_sequences)
output1 = Dense(NUM_EMOTIONS, activation='softmax')(x)
output2 = Dense(1, activation='sigmoid')(x)

model =  Model(input=[sequence_input], output=[output1, output2])
model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'],
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

model.fit([(x1_train, y1_train), (x2_train, y2_train)],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          shuffle=True,
          validation_data=([(x1_test, y1_test), (x2_test, y2_test)]))

score, acc = model.evaluate([(x1_test, y1_test), (x2_test, y2_test)],
                            batch_size=BATCH_SIZE)

print('\nTest score:', score)
print('Test accuracy:', acc)

print('Write word vectors to', OUTPUT_MODEL)
with open(OUTPUT_MODEL, 'w') as f:
    f.write(" ".join([str(V), str(EMBEDDING_DIM)]))
    f.write("\n")

    vectors = model.get_weights()[0]  # shape: V x EMBEDDING_DIM

    for word, i in emo_word_to_index.items():
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vectors[i, :]))))
        f.write("\n")

    for word, i in sent_word_to_index.items():
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vectors[i, :]))))
        f.write("\n")
