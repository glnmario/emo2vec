from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from pickle import dump

RESOURCES_PATH = 'resources/'
CORPUS_PATH = RESOURCES_PATH + 'twitter_corpus.txt'
LEXICON_PATH = RESOURCES_PATH + 'emolex.txt'
LABELS_PATH = RESOURCES_PATH + 'labels.p'
NUM_EMOTIONS = 8
MAX_SEQUENCE_LENGTH = 50


def normalize(array):
    norm = np.add.reduce(array)
    if norm != 0:
        array = [(x / norm) for x in array]
    return array


def read_emo_lemma(aline):
    """
    Splits a line into lemma l, emotion e, and l(e).
    l(e) := 1 if lemma l has emotion e according to the lexicon
    l(e) := 0 otherwise
    """
    split = aline.split()
    return split[0], split[1], int(split[2])


def build_fuzzy_lexicon(lexicon_path):
    """
    Based on the emotion lexicon, create a mapping from an emotion word to its label probability distribution
    """
    lexicon = dict()
    with open(lexicon_path, 'r') as f:
        emo_idx = 0  # anger: 0, anticipation: 1, disgust: 2, fear: 3, joy: 4, sadness: 5, surprise: 6, trust: 7

        for l in f:
            lemma, emotion, has_emotion = read_emo_lemma(l)

            if emotion == 'anger':  # i.e. if lemma not in lexicon.keys()
                lexicon[lemma] = np.empty(shape=(NUM_EMOTIONS,))
            if emotion == 'positive' or emotion == 'negative':
                continue

            lexicon[lemma][emo_idx] = has_emotion

            if emo_idx < NUM_EMOTIONS - 1:
                emo_idx += 1
            else:
                # normalize: emotion-label probabilities for a lemma should sum up to 1
                lexicon[lemma] = normalize(lexicon[lemma])
                # reset index - next line contains a new lemma
                emo_idx = 0
    return lexicon

print('Create mapping: emotion word -> label probability distribution.')
prob_lexicon = build_fuzzy_lexicon(LEXICON_PATH)

print('Read and tokenize corpus.')
texts = []

with open(CORPUS_PATH, 'r') as f:
    for line in f:
        line_split = line[20:].split(sep='\t:: ')
        texts.append(line_split[0].strip())

    print('Found %s texts.' % len(texts))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input
sequences = pad_sequences(sequences, MAX_SEQUENCE_LENGTH, padding='post')

# dictionary mapping an index to the word it represents in the corpus (invert word->index mapping as it is bijective)
index_to_word = {i: w for w, i in tokenizer.word_index.items()}

fuzzy_labels = []  # shape=(len(sequences), NUM_EMOTIONS)) -- label probability distribution for sequences

print('Label the texts.')
for seq in sequences:
    seq_labels = np.zeros(shape=(MAX_SEQUENCE_LENGTH, NUM_EMOTIONS))
    j = 0  # index of token in a sequence (different from token_id)

    for token_id in seq:
        if token_id == 0:  # we reached the padding zeros
            break
        token = index_to_word[token_id]
        if token in prob_lexicon.keys():
            seq_labels[j] += prob_lexicon[token]
        j += 1

    fuzzy_labels.append(normalize(np.add.reduce(seq_labels)))

print('Write %d labels to %s' % (len(fuzzy_labels), LABELS_PATH))
dump(fuzzy_labels, open(LABELS_PATH, 'wb'))
