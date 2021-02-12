import nltk
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
    pass
