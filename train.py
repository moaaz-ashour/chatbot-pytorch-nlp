import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

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
all_words = [stem(word) for word in all_words if word not in ignored_characters]
# remove duplicates and sort the list
all_words = sorted(set(all_words))

# create bag of words
x_train = []  # for bag of words
y_train = []  # associated number for each tag

# iterate over (patterns, tags)
for (tokenized_pattern, tag) in xy:
    bag = bag_of_words(tokenized_pattern, all_words)
    x_train.append(bag)

    # get index of each tag (label them) and append to y_train
    label = tags.index(tag)
    y_train.append(label)  # we want only the class labels (cross-entropy loss)

# convert the training data to numpy array
X_train = np.array(x_train)
y_train = np.array(y_train)


# create new Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        
        # data
        self.x_data = X_train    
        self.y_data = y_train

    def __getitem__(self, idx): 
        return self.x_data[idx], self.y_data[idx]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()

    
# Hyperparameters:
batch_size = 8
input_size = len(X_train[0]) #> first bag of words length > all_words
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

def main():
    # check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create DataLoader
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # create model and push it to device if available
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # loss and optimzer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return device, train_loader, model, criterion, optimizer

def training(device, train_loader, model, criterion, optimizer):
    # training loop
    for epoch in range(num_epochs):
        # use training loader
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            # forward pass - words from training loader as input
            outputs = model(words)

            # calculate loss
            loss = criterion(outputs, labels.long())

            # backward pass and optimizer step
            # 1. empty the gradient
            optimizer.zero_grad()
            # 2.1 calculate back propagation  
            loss.backward()
            # 2.2 parameter update based on parameter.grad using internally stored grad to update parameters values.
            optimizer.step()

        if (epoch +1) % 100 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

    print(f"final loss={loss.item():.4f}")

# storing data
def save_data(model, optimizer):
    data = {
        "model_state_dict": model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)
    print(f"training complete. File saved to {FILE}")


if __name__ == '__main__':
    device, train_loader, model, criterion, optimizer = main()
    training(device, train_loader, model, criterion, optimizer)
    save_data(model, optimizer)