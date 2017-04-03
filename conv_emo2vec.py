from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, Conv1D, MaxPooling1D, Dense, Flatten
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
np.random.seed(13)


TRAIN_OVER_TEST = 0.7
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 150
EPOQUES = 50
RESOURCES_PATH = 'resources/'
CORPUS_PATH = RESOURCES_PATH + 'twitter_corpus.txt'
MODEL_PATH = RESOURCES_PATH + 'vectors.txt'

labels_index = {'anger': 0,
                'anticipation': 1,
                'disgust': 2,
                'fear': 3,
                'joy': 4,
                'sadness': 5,
                'surprise': 6,
                'trust': 7}

print('Indexing word vectors.')

embeddings_index = {}
f = open(MODEL_PATH, 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

# texts[i] has emotion_labels[i]
texts = []
emotion_labels = []

with open(CORPUS_PATH, 'r') as f:
    for line in f:
        line_split = line[20:].split(sep='\t:: ')
        texts.append(line_split[0].strip())
        emotion_labels.append(labels_index[line_split[1].strip()])

    print('Found %s texts.' % len(texts))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input

word_to_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)
V = len(word_to_index)  # vocabulary size

print('Found %s unique tokens.' % V)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

emotion_labels = to_categorical(np.asarray(emotion_labels))
print('Shape of data tensor:', data.shape)  # (21051, 25)
print('Shape of label tensor:', emotion_labels.shape)  # (21051, 7)   WHY SEVEN?

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

print('Preparing embedding matrix.')

embedding_matrix = np.zeros((V + 1, EMBEDDING_DIM))

# prepare embedding matrix
num_words = min(MAX_SEQUENCE_LENGTH, V)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_to_index.items():
    if i >= MAX_SEQUENCE_LENGTH:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = True so as to let the embeddings vary
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test))
