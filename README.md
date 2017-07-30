# emo2vec

#### Learning emotion-specific embeddings.

In the proposed framework, emotion-specific word embeddings are learned from a corpus of texts labeled with six basic emotions (anger, disgust, fear, joy, sadness, and surprise). We use a Long Short Term Memory (LSTM) recurrent network that learns emotion-specific representations of words via backpropagation, where the _emotion-specificity_ of a word vector refers to the ability to encode affectual orientation and strength in a subset of its dimensions.

The derived vector space model is used to expand an existing emotion lexicon via a novel variant of the Label Propagation algorithm that is tailored to distributed word representations. Batch gradient descent is used to accelerate the optimization of label propagation and to make the optimization feasible for large graphs. 


#### Resources

- **NRC Word-Emotion Association Lexicon** aka EmoLex: association of words with eight emotions (anger, fear, anticipation, trust, surprise, sadness, joy, and disgust) and two sentiments (negative and positive) manually annotated on Amazon's Mechanical Turk. The English version of this lexicon is used in emo2vec, it contains 14,182 unigrams.

    - Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney, Computational Intelligence, 29 (3), 436-465, 2013.

    - Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an Emotion Lexicon, Saif Mohammad and Peter Turney, In Proceedings of the NAACL-HLT 2010 Workshop on Computational Approaches to Analysis and Generation of Emotion in Text, June 2010, LA, California.

- **Hashtag Emotion Corpus**: a corpus of 21051 Twitter posts labeled with eight emotions using emotion-word hashtags.

    - Using Hashtags to Capture Fine Emotion Categories from Tweets. Saif M. Mohammad, Svetlana Kiritchenko, Computational Intelligence, in press.

    - \#Emotional Tweets, Saif Mohammad, In Proceedings of the First Joint Conference on Lexical and Computational Semantics (*Sem), June 2012, Montreal, Canada.
    
Both available at http://saifmohammad.com/WebPages/lexicons.html.
