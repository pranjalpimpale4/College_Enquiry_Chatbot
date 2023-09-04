import os
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from googletrans import Translator

translater = Translator()
from gtts import gTTS
from playsound import playsound
import pyttsx3

engine = pyttsx3.init()

from tempfile import TemporaryFile, NamedTemporaryFile

# from pygame import mixer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# from "C:/Users/sanchit/Desktop" import data2.pth
with open('databasefinal2.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data2.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "College Enquiry"
hindi_bot_name = translater.translate(bot_name, dest='hi')
print("Hello, Welcome to SPIT")


def reply(text):
    sentence = text
    if sentence == "quit":
        return "bye"

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    i = 0
    # temp = TemporaryFile()
    # temp = NamedTemporaryFile(suffix='.mp3').name
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                # print(tag)
                i = i + 1
                reply = random.choice(intent['responses'])
                return reply
                # print(f"{bot_name}: {reply}")
                # e_obj = gTTS(text=reply, slow=False, lang='en')
                # e_obj.save('eng.mp3')
                # playsound('eng.mp3')

                # out = translater.translate(reply, dest='hi')
                # return out
                # print(f"{hindi_bot_name.text}: {out.text} \n")
                # print('\n')
                # mixer.init()
                # obj = gTTS(text=out.text, slow=False, lang='hi')
                # if not os.path.isfile('hindi.mp3'):
                # obj.save('hindi.mp3')
                #     print("present")
                # obj.write_to_fp(temp)
                # temp.seek(0)
                # playsound(temp)
                # mixer.music.load(temp)
                # mixer.music.play()
                # temp.close()
                # obj.write_to_fp('hindi.mp3')
                # playsound('hindi.mp3')
                # os.remove('hindi.mp3')

                # engine.setProperty('rate', 150)
                # engine.say(reply)
                # engine.runAndWait()
                #
                # engine.say(out.text)
                # engine.runAndWait()
    else:
        return "Sorry, I do not understand. Repeat Again"
        # print(f"{bot_name}: I do not understand...")
