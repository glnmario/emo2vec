import numpy as np
from pickle import load
from gensim.models.keyedvectors import KeyedVectors

RESOURCES_PATH = 'resources/'
CORPUS_PATH = RESOURCES_PATH + 'twitter_corpus.txt'
LEXICON_PATH = RESOURCES_PATH + 'emolex.txt'
LABELS_PATH = RESOURCES_PATH + 'labels.p'
MODEL_PATH = RESOURCES_PATH + 'vectors.txt'
NUM_SAMPLES = 10000

label_to_index = {'anger': 0,
                  'anticipation': 1,
                  'disgust': 2,
                  'fear': 3,
                  'joy': 4,
                  'sadness': 5,
                  'surprise': 6,
                  'trust': 7}


def agreement(gold, other):
    n = len(gold)
    assert (n == len(other))
    assert (n > 0)

    count_agree = 0

    for i in range(n):
        if type(other[i]) is int:
            count_agree += int(gold[i] == other[i])
        elif type(other[i]) is list:
            count_agree += int(gold[i] in other[i])  # allows fuzzy labels

    return count_agree / n


# Extract gold standard labels
gold_labels = []
with open(CORPUS_PATH, 'r') as f:
    for line in f:
        line_split = line[20:].split(sep='\t:: ')
        gold_labels.append(label_to_index[line_split[1].strip()])
N = len(gold_labels)

# Load fuzzy labels obtained with trivial annotation (simple_classifier.py)
fuzzy_labels = load(open(LABELS_PATH, 'rb'))
trivial_labels = []

# Assign labels with maximum probability
for label_distribution in fuzzy_labels:
    max_prob = np.amax(label_distribution)
    if max_prob == 0:
        trivial_labels.append(-1)  # no emotion detected
    else:
        # all emotions with max probability value
        trivial_labels.append([j for j, k in enumerate(label_distribution) if k == max_prob])

# Generate (pseudo-)random labels
random_labels = []
for x in range(N):
    random_labels.append(np.random.randint(0, 8))

# Compute agreement
random_agreement = 0
for i in range(NUM_SAMPLES):
    random_agreement += agreement(gold_labels, random_labels)

random_agreement /= NUM_SAMPLES

print('Random labels agreement (mean of {0} samples): {1}'.format(NUM_SAMPLES, random_agreement))
print('Trivial labels agreement: {}'.format(agreement(gold_labels, trivial_labels)))


w2v = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)

print(w2v.most_similar(positive=['surprise']))
print(w2v.most_similar(positive=['happy']))
print(w2v.most_similar(positive=['angry']))
print(w2v.most_similar(positive=['disgusted']))