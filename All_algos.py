import json

from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

X_train = []
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    Y_train.append(label)  # CrossEntropyLoss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)


forest = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
Y_pred_train = forest.predict(X_train)
rf_train_acc = accuracy_score(Y_train, Y_pred_train)
print(f'Training Accuracy for Random Forest Algorithm :  { rf_train_acc*100 :.2f}')
# print("Training Accuracy for Random Forest Algorithm : " + str(accuracy_score(Y_train, Y_pred_train)))
Y_pred_test = forest.predict(X_test)
rf_test_acc = accuracy_score(Y_test, Y_pred_test)
print(f'Testing Accuracy for Random Forest Algorithm :  { rf_test_acc*100 :.2f}')
# print("Testing Accuracy for Random Forest Algorithm : " + str(accuracy_score(Y_test, Y_pred_test)))
print("\n")

Decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Decision_tree.fit(X_train, Y_train)
Y_pred_train = Decision_tree.predict(X_train)
dt_train_acc = accuracy_score(Y_train, Y_pred_train)
print(f'Training Accuracy for Decision Tree Algorithm :  { dt_train_acc*100 :.2f}')
# print("Training Accuracy for Decision Tree Algorithm : " + str(accuracy_score(Y_train, Y_pred_train)))
Y_pred_test = Decision_tree.predict(X_test)
dt_test_acc = accuracy_score(Y_test, Y_pred_test)
print(f'Testing Accuracy for Decision Tree Algorithm :  { dt_test_acc*100 :.2f}')
# print("Testing Accuracy for Decision Tree Algorithm : " + str(accuracy_score(Y_test, Y_pred_test)))
print("\n")

Naive_Bayes = GaussianNB()
Naive_Bayes.fit(X_train, Y_train)
Y_pred_train = Naive_Bayes.predict(X_train)
nb_train_acc = accuracy_score(Y_train, Y_pred_train)
print(f'Training Accuracy for Naive Bayes Algorithm :  { nb_train_acc*100 :.2f}')
# print("Training Accuracy for Naive_Bayes Algorithm : " + str(accuracy_score(Y_train, Y_pred_train)))
Y_pred_test = Naive_Bayes.predict(X_test)
nb_test_acc = accuracy_score(Y_test, Y_pred_test)
print(f'Training Accuracy for Naive Bayes Algorithm :  { nb_test_acc*100 :.2f}')
# print("Testing Accuracy for Naive_Bayes Algorithm : " + str(accuracy_score(Y_test, Y_pred_test)))
print("\n")