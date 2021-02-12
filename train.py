import json
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []  # will hold tokenized patterns and tags

# loop over intents
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokenized_pattern = tokenize(pattern)
        all_words.extend(tokenized_pattern)  # extend with array of words
        xy.append((tokenized_pattern, tag))
ignored_characters = ['?', "!", '.', ',']

# apply stemming to tokenized all_words list and excluding ignored_words
all_words = [stem(word) for word in all_words if word not in ignored_words]
# remove duplicates and sort the list
all_words = sorted(set(all_words))