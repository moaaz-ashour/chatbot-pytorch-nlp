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


input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state_dict = data['model_state_dict']

# create model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
# load state dict 
model.load_state_dict(model_state_dict)
# set to evaluation mode
model.eval()


# implement ChatBot
bot_name = "Chat Bot"
print("Let's Chat! type 'quit' to exit")

While True:
    sentence = input("You: ")
    if sentence == "quit":
        break