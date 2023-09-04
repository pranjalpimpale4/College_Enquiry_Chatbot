import json
# from torch import dtype
from torchsummary import summary

from torch.cuda import is_available
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn  as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

with open('databasefinal2.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',','\\n',':']
all_words = [stem(w) for w in all_words if w not in ignore_words]
#print(all_words)
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print("Total number of labels = " + str(len(tags)))
print(tags)