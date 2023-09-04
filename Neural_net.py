import json
# from torch import dtype
from torchsummary import summary

from torch.cuda import is_available
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

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

ignore_words = ['?', '!', '.', ',', '\\n', ':']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# print(all_words)
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

X = []
Y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)

    label = tags.index(tag)
    Y.append(label)  # CrossEntropyLoss

X = np.array(X)
Y = np.array(Y)


class ChatDataset(Dataset):
    def __init__(self, X, Y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = Y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# HyperParameters
batch_size = 120
hidden_size = 120
output_size = len(tags)
input_size = len(X[0])
print(input_size, len(all_words))
print(output_size, tags)
learning_rate = 0.0004
num_epochs = 2500
# def run():
#    torch.multiprocessing.freeze_support()
#    print('loop')

# if __name__ == '__main__':
#    run()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

training_dataset = ChatDataset(X_train,Y_train)
train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

testing_dataset = ChatDataset(X_test, Y_test)
test_loader = DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
# test_loader = DataLoader(dataset=X_test, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

print(model)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_acc = 0

print("\n\nModel Training: ")

for epoch in range(num_epochs):
    # epoch_acc = 0
    correct = 0
    n_samples = 0
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # print(f'Labels = {len(labels)}')

        # forward
        outputs = model(words)
        # print(f'outputs = {len(outputs)}')
        loss = criterion(outputs, labels)

        # backward & optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # rounded_preds = torch.round(outputs)
        # #print(rounded_preds.shape)
        # #print(labels.shape)
        _, pred_label = torch.max(outputs, 1)
        # #print(pred_label.shape)
        n_samples += labels.shape[0]
        correct += (pred_label == labels).sum().item()
        # #print(f'Correct length = {len(correct)} and Correct sum = {correct.sum()}')
        # acc = correct.sum() / len(outputs)

        # epoch_acc += acc

    if (epoch + 1) % 100 == 0:
        acc = 100.0 * correct / n_samples
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}, Accuracy={acc:.2f}')
        if((epoch+1)!=num_epochs):
            acc = 0
        # epoch_acc = 0
        # print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}, Accuracy={acc:.2f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

valid_loss = 0
print(model.eval())
correct = 0
n_samples = 0
for (words, labels) in test_loader:
    words = words.to(device)
    labels = labels.to(dtype=torch.long).to(device)
    # Forward Pass
    target = model(words)
    # Find the Loss
    loss = criterion(target, labels)
    # Calculate Loss
    valid_loss += loss.item()
    _, pred_label = torch.max(target, 1)
    n_samples += labels.shape[0]
    correct += (pred_label == labels).sum().item()
acc = 100.0 * correct / n_samples
print(f'NeuralNet Testing , loss={loss.item():.4f}, Accuracy={acc:.2f}')

FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete, file saved to {FILE}')