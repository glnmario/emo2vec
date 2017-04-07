from gensim.models.keyedvectors import KeyedVectors
from numpy import loadtxt

RESOURCES_PATH = 'resources/'
TARGET_WORDS_PATH = RESOURCES_PATH + 'target_words.txt'
MODEL_PATHS = (RESOURCES_PATH + p for p in ['vectors.txt',
                                            'cnn_vectors.txt',
                                            'lstm_vectors.txt',
                                            'cnn_lstm_vectors.txt'])

with open(TARGET_WORDS_PATH, 'r') as f:
    target_words = f.read().split('\n')

for path in MODEL_PATHS:
    w2v = KeyedVectors.load_word2vec_format(path, binary=False)
    print('Model: ', path)
    for word in target_words:
        print(word, ' ', w2v.most_similar(positive=[word]))
    print("-"*50)
