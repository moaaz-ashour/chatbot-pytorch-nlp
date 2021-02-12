import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# 1
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# 2
def stem(word):
    return stemmer.stem(word.lower())


# 3
def bag_of_words(tokenized_sentence, all_words):
    # apply stemming
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    # initialize a bag with zeros for all words
    bag = np.zeros(len(all_words), dtype=np.float32)
    # iterate over all words. If word is found, change to 1.0
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0 
    return bag