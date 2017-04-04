# emo2vec
Learning emotion-specific embeddings.


#### Code

- `skipgram_ns.py` is a [Keras](https://keras.io/) implementation of the skip-gram model
- `cnn_emo2vec.py` is a 1D convolutional neural network with pooling used to learn emotion-specific embeddings
- `lstm_emo2vec.py` is a LSTM recurrent neural network used to learn emotion-specific embeddings
- `cnn_lstm_emo2vec.py` is the combination of CNN and LSTM
- `simple_classifier.py` is a trivial fuzzy classifier based on Emolex
- `agreement.py` evaluates agreement between gold standard and otherwise obtained labels
- `similarity.py` prints the 10 most similar word-vectors to a list of defined target words, according to four different models: skipgram with negative sampling, cnn-emo2vec, lstm-emo2vec, cnn_lstm-emo2vec


#### Resources

- **NRC Word-Emotion Association Lexicon** aka EmoLex: association of words with eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive) manually annotated on Amazon's Mechanical Turk. The English version of this lexicon is used in emo2vec, it contains 14,182 unigrams.

    - Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney, Computational Intelligence, 29 (3), 436-465, 2013.

    - Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon, Saif Mohammad and Peter Turney, In Proceedings of the NAACL-HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010, LA, California.

- **Hashtag Emotion Corpus**: a corpus of 21051 Twitter posts labeled with eight emotions using emotion-word hashtags.

    - Using Hashtags to Capture Fine Emotion Categories from Tweets. Saif M. Mohammad, Svetlana Kiritchenko, Computational Intelligence, in press.

    - \#Emotional Tweets, Saif Mohammad, In Proceedings of the First Joint Conference on Lexical and Computational Semantics (*Sem), June 2012, Montreal, Canada.
    
Both available at http://saifmohammad.com/WebPages/lexicons.html.
