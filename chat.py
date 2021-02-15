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

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    # tokenize sentence:
    tok_sentence = tokenize(sentence)
    # create an array of bag of words with tokenized sentence and words from the saved file
    x = bag_of_words(tok_sentence, all_words)
    # reshapeing to 1 row (i.e. 1 sample) and 1 column [54]
    x = x.reshape(1, x.shape[0])
    # convert it to torch tensor
    x = torch.from_numpy()

    # pass data as model input to get predictions
    output = model(x)
    # getting predictions
    _, predicted = torch.max(output, dim=1)
    # get actual tag
    tag = tags[predicted.item()] # tags[class_label_number], e.g. greeting
    # find corresponding intent by looping over intents and check if tags matches
    for intent in intents["intents"]:
        if tag == intent["tag"]:
            # possible response
            print(f"{bot_name}: {random.choice(intent["responses"])}")
