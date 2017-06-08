from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np

CORPUS_PATH = 'resources/data/hashtag/hashtag_corpus.txt'

labels_index = {'anger': 0,
                'disgust': 1,
                'fear': 2,
                'joy': 3,
                'sadness': 4,
                'surprise': 5}
NUM_EMOTIONS = len(labels_index)

# texts[i] has emotion_labels[i]
texts = []
emotion_labels = []

print('Process corpus...')
with open(CORPUS_PATH, 'r', encoding='UTF-8') as f:
    for line in f:
        line_split = line[20:].split(sep="\t:: ")
        texts.append(line_split[0].strip())
        emotion_labels.append(labels_index[line_split[1].strip()])
print('Found %s texts.' % len(texts))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input
max_seq_len = np.max([len(s) for s in sequences])

word_to_index = tokenizer.word_index  # dictionary mapping words (str) to their rank/index (int)

data = pad_sequences(sequences, maxlen=max_seq_len)
emotion_labels = to_categorical(np.asarray(emotion_labels), NUM_EMOTIONS)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', emotion_labels.shape)

np.save('resources/data/hashtag/data.npy', data)
np.save('resources/data/hashtag/labels.npy', emotion_labels)
np.save('resources/data/hashtag/word_index.npy', word_to_index)


def split(train, val=0.):
    """
    :param train: percentage of training set size (e.g. 0.6)
    :param val: percentage of training set size (e.g. 0.2)
    :return: x_train, y_train, x_val, y_val, x_test, y_test
    """
    np.random.seed(13)

    texts = np.load('resources/data/hashtag/data.npy')
    labels = np.load('resources/data/hashtag/labels.npy')

    # split the data into a training set and a validation set
    indices = np.arange(texts.shape[0])
    np.random.shuffle(indices)
    texts = texts[indices]
    labels = labels[indices]
    num_train_samples = int(train * data.shape[0])
    num_valid_samples = int(val * data.shape[0])

    x_train = texts[:num_train_samples]
    y_train = labels[:num_train_samples]
    x_val = texts[num_train_samples: num_train_samples + num_valid_samples]
    y_val = labels[num_train_samples: num_train_samples + num_valid_samples]
    x_test = texts[num_train_samples + num_valid_samples:]
    y_test = labels[num_train_samples + num_valid_samples:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def word_index():
    return np.load('resources/data/hashtag/word_index.npy').item()

def max_sequence_len():
    texts = np.load('resources/data/hashtag/data.npy')
    return texts.shape[1]
