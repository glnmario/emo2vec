from __future__ import print_function
from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """

    RESOURCES_PATH = 'resources/'
    CORPUS_PATH = RESOURCES_PATH + 'twitter_corpus.txt'
    PRETRAINED_MODEL = RESOURCES_PATH + 'vectors.txt'
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 36
    TRAIN_OVER_TEST = 0.7
    numpy.random.seed(13)

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

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    emotion_labels = to_categorical(numpy.asarray(emotion_labels), NUM_EMOTIONS)

    # split the data into a training set and a validation set
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    emotion_labels = emotion_labels[indices]
    num_train_samples = int(TRAIN_OVER_TEST * data.shape[0])

    x_train = data[:num_train_samples]
    y_train = emotion_labels[:num_train_samples]
    x_test = data[num_train_samples:]
    y_test = emotion_labels[num_train_samples:]

    print('Indexing word vectors.')

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = True so as to let the embeddings vary
    embeddings_index = {}
    with open(PRETRAINED_MODEL, 'r') as f:
        next(f)
        for line in f:
            values = line.split()
            if len(values) != EMBEDDING_DIM + 1:  # probably an error occured during tokenization
                continue
            word = values[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    print('Preparing embedding matrix.')
    embedding_matrix = numpy.zeros((V + 1, EMBEDDING_DIM))
    for word, i in word_to_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i - 1] = embedding_vector

    return x_train, y_train, x_test, y_test, embedding_matrix, V


def model(x_train, y_train, x_test, y_test, embedding_matrix, V):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """

    NUM_EMOTIONS = 8
    EMBEDDING_DIM = 300
    MAX_SEQUENCE_LENGTH = 36

    embedding_layer = Embedding(V + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = LSTM({{choice([64, 128, 256])}}, dropout={{uniform(0, 0.4)}}, recurrent_dropout={{uniform(0, 0.4)}})(embedded_sequences)
    preds = Dense(NUM_EMOTIONS, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'adagrad'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=1,
              verbose=2,
              shuffle=True,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials(),
                                          rseed=13)

    X_train, Y_train, X_test, Y_test, _, _ = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
