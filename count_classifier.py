from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np


n_classes = 6


def l1_normalize(vector):
    norm = np.sum(vector)
    if norm == 0:
        return vector  # zeros vector
    return vector / norm


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
                lexicon[lemma] = np.empty(shape=(n_classes,))
            if emotion == 'positive' or emotion == 'negative':
                continue

            lexicon[lemma][emo_idx] = has_emotion

            if emo_idx < n_classes - 1:
                emo_idx += 1
            else:
                # normalize: emotion-label probabilities for a lemma should sum up to 1
                lexicon[lemma] = l1_normalize(lexicon[lemma])
                # reset index - next line contains a new lemma
                emo_idx = 0
    return lexicon


def classify(corpus_path, lexicon_path):
    """
    Return a list of probability distributions.
    """
    # Create mapping: emotion word -> label probability distribution
    prob_lexicon = build_fuzzy_lexicon(lexicon_path)

    print('Read and tokenize corpus.')
    texts = []
    with open(corpus_path, 'r') as f:
        for line in f:
            line_split = line[20:].split(sep='\t:: ')
            texts.append(line_split[0].strip())

        print('Found %s texts.' % len(texts))


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)  # one sequence of tokens per text input
    max_seq_len = np.max([len(s) for s in sequences])
    sequences = pad_sequences(sequences, max_seq_len, padding='post')

    # Dictionary mapping an index to the word it represents in the corpus (invert word->index mapping as it is bijective)
    index_to_word = {i: w for w, i in tokenizer.word_index.items()}

    # Label probability distribution for sequences, shape=(len(sequences), n_classes)
    fuzzy_labels = []

    print('Label the texts.')
    for seq in sequences:
        seq_labels = np.zeros(shape=(max_seq_len, n_classes))
        j = 0  # index of token in a sequence (different from token_id)

        for token_id in seq:
            if token_id == 0:  # we reached the padding zeros
                break
            token = index_to_word[token_id]
            if token in prob_lexicon.keys():
                seq_labels[j] += prob_lexicon[token]
            j += 1

        labels = l1_normalize(np.sum(seq_labels, 0))
        fuzzy_labels.append(labels)

    return fuzzy_labels
