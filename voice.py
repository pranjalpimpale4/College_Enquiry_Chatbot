import pyttsx3
import speech_recognition as sr

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        r.pause_threshold = 0.8
        r.energy_threshold = 200
        audio = r.listen(source)

    try:
        print("Recognizing..")
        # query = r.recognize_google(audio, language='en-in')
        query = r.recognize_google(audio, language='hi-in')
        print(f"User said: {query}\n")
    except Exception as e:
        # print(e)
        print("Say that again please...")
        return "None"
    return query

if __name__=="__main__":
    while True:
        takeCommand()
