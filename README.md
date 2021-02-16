# Chatbot with PyTorch and NLP :rocket:



This is a minimal chatbot project using PyTorch, numpy and NLP techniques.
#### For project board click [here](https://github.com/users/moaaz-ashour/projects/3).
---

## Quick Links
* [Introduction](#introduction)
* [Installation Guide](#installation-guide)

## Introduction

Some of the functionalities you would see in this project: <br>
    * Define conversational intents as tags, patterns and responses. <br>
    * NLP technicques (NLTK) for tokenizing, stemming, creation of bag-of-words and data cleaning. <br>
    * Creating, saving and loading PyTorch model. <br>
    * Creating PyTorch Dataset from Training Data. <br>
    * Implementation of feedforward Neural Network in PyTorch with 2 hidden layers. <br>
    * Neural Network training (DataLoader, forward and backward passes, calculating the loss, back propagation and parameters update. <br>
    * Applying `softmax` to get actual probabilities for predicted tags. <br>

--- 

## Setup Guide

To set up the project, follow these steps:<br>
   1. Create a project directory by going to Terminal and typing `mkdir project_name`. 
   2. Type `cd project_name`
   3. Now let's install, create, and activate a new virtual environment.
   
   - Install `virtualenv` package.
   - Create virtual environment: 
      - On macOS and Linux: `python3 -m venv env`.
      - On Windows: `py -m venv env`
   - Activate virtual environment:
      - On macOS and Linux: `source env/bin/activate`
      - On Windows: `.\env\Scripts\activate`
   4. Using [requirements.txt](https://github.com/moaaz-ashour/chatbot-pytorch-nlp/blob/master/requirements.txt) file to install project-required packages:
      - `python3 -m pip install -r requirements.txt`
    
   5. Now, in terminal, run: 
        - `python -m train.py` (might take a while)
        - `python -m chat.py`


:tada: :tada: :tada:

You might need to modify [intents.json](https://github.com/moaaz-ashour/chatbot-pytorch-nlp/blob/master/intents.json) to add or remove intents. Remember that if you do so, you will have to rerun training.

Happy Chatbotting :robot:




