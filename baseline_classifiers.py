from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np

from count_classifier import classify


corpus_path = 'resources/data/hashtag/hashtag_corpus.txt'
lexicon_path = 'resources/data/emolex.txt'
n_samples = 10000

label_to_index = {'anger': 0,
                  'disgust': 1,
                  'fear': 2,
                  'joy': 3,
                  'sadness': 4,
                  'surprise': 5}


# Extract gold standard labels
gold_labels = []
with open(corpus_path, 'r', encoding='UTF8') as f:
    for line in f:
        line_split = line[20:].split(sep='\t:: ')
        gold_labels.append(label_to_index[line_split[1].strip()])
N = len(gold_labels)
gold_labels = np.asarray(gold_labels)


# Obtain fuzzy labels from count-based classifier (simple_classifier.py)
fuzzy_labels = classify(corpus_path, lexicon_path)
trivial_labels = []

# Assign labels with maximum probability
for label_distr in fuzzy_labels:
    max_label = np.random.choice(np.where(label_distr == label_distr.max())[0])
    trivial_labels.append(max_label)  # no emotion detected

trivial_labels = np.array(trivial_labels)

print("Naive count-based emotion classifier")
print("P: {}  R: {}  F1: {}".format(precision_score(gold_labels, trivial_labels, average='micro'),
                                    recall_score(gold_labels, trivial_labels, average='micro'),
                                    f1_score(gold_labels, trivial_labels, average='micro')))


# Generate (pseudo-)random labels
random_labels = np.random.randint(0, 6, size=(N))

pr = 0
rec = 0
f1 = 0
for i in range(n_samples):
    pr += precision_score(gold_labels, random_labels, average='micro')
    rec += recall_score(gold_labels, random_labels, average='micro')
    f1 += f1_score(gold_labels, random_labels, average='micro')

print("Random classifier")
print("P: {}  R: {}  F1: {}".format(pr / n_samples,
                                    rec / n_samples,
                                    f1 / n_samples))
