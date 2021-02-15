import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


# check GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)