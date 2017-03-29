from keras.preprocessing.sequence import skipgrams
from keras.models import Sequential
from keras.layers import Embedding, Reshape, Activation, Merge
from keras.preprocessing.text import Tokenizer
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
np.random.seed(13)


TRAIN_OVER_TEST = 0.7
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300
EPOQUES = 50
CORPUS_PATH = '../twitter_corpus.txt'
MODEL_PATH = 'vectors.txt'

labels_index = {'anger': 0,
                'anticipation': 1,
                'disgust': 2,
                'fear': 3,
                'joy': 4,
                'sadness': 5,
                'surprise': 6,
                'trust': 7}

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

# target
target = Sequential()
target.add(Embedding(V + 1, EMBEDDING_DIM, input_length=1))

# context
context = Sequential()
context.add(Embedding(V + 1, EMBEDDING_DIM, input_length=1))

model = Sequential()
# dot product between a word embedding and a context embedding can be used to train Mikolov-style word embeddings
model.add(Merge([target, context], mode='dot', dot_axes=2))
model.add(Reshape((1,), input_shape=(1, 1)))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="rmsprop")

for _ in range(EPOQUES):
    loss = 0.
    for seq in sequences:
        data, labels = skipgrams(sequence=seq, vocabulary_size=V + 1, window_size=5, negative_samples=5.)
        X = [np.array(x) for x in zip(*data)]
        Y = np.array(labels, dtype=np.int32)
        if X:
            loss += model.train_on_batch(X, Y)
    print(loss)

with open(MODEL_PATH, 'w') as f:
    f.write(" ".join([str(V), str(EMBEDDING_DIM)]))
    f.write("\n")

    vectors = model.get_weights()[0][0]  # shape: V x EMBEDDING_DIM  (e.g. V x 300)

    for word, i in word_to_index.items():
        f.write(word)
        f.write(" ")
        f.write(" ".join(map(str, list(vectors[i, :]))))
        f.write("\n")

w2v = KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)

print(w2v.most_similar(positive=['surprise']))
print(w2v.most_similar(positive=['happy']))
print(w2v.most_similar(positive=['angry']))
print(w2v.most_similar(positive=['disgusted']))
