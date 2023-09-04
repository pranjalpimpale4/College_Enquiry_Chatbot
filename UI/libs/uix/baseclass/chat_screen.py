from kivy.animation import Animation
from kivy.properties import DictProperty, ListProperty, StringProperty

from UI.libs.uix.components.boxlayout import PBoxLayout
from UI.libs.uix.components.dialog import PDialog
from UI.libs.uix.components.screen import PScreen
from UI.libs.uix.components.toast import toast

from reply import reply

import speech_recognition as sr
from googletrans import Translator
translator = Translator()

# from os import environ
# environ['KIVY_TEXT'] = 'sdl2'

from gtts import gTTS
from kivy.core.audio import SoundLoader
from UI.voice_out import voice_out
# import pyttsx3
# engine = pyttsx3.init()
from playsound import playsound

class ChatScreen(PScreen):
    user = DictProperty()
    title = StringProperty()
    chat_logs = ListProperty()
    flag = False

    # # def __init__(self):
    #     self.lang = ''

    def lang_detect(self, language):
        self.lang = language
        self.flag = True
        print(language)

    def send(self, text):
        if not text:
            toast("Please enter any text!")
            return

        if self.flag is False:
            toast("Please Select language!")
            # print(self.lang)
            return
        elif self.lang == 'hi-in':
            # print(f"{hindi_bot_name.text}: {out.text} \n")
            self.chat_logs.append(
                {"text": text, "send_by_user": True, "pos_hint": {"right": 1}}
            )
            rep_hindi = translator.translate(reply(text), dest='hi')
            print(type(rep_hindi))
            print(rep_hindi.text)
            self.chat_logs.append(
                {
                    "text": rep_hindi.text,
                    "send_by_user": False,
                }
            )

            voice_out(rep_hindi.text, 'hi-in')
            # hindi_voice = gTTS(rep_hindi.text, slow=False, lang='hi')
            # hindi_voice.save('hindi.mp3')
            # op_voice = SoundLoader.load('hindi.mp3')
            # op_voice.play()

        else:
            self.chat_logs.append(
                {"text": text, "send_by_user": True, "pos_hint": {"right": 1}}
            )
            self.chat_logs.append(
                {
                    "text": reply(text),
                    "send_by_user": False,
                }
            )

            voice_out(reply(text), 'en-in')

            # eng_voice = gTTS(reply(text), slow=False, lang='en')
            # eng_voice.save('eng.wav')
            # print("saved")
            # # engine.setProperty('rate', 150)
            # # engine.say(reply)
            # # engine.runAndWait()
            # # playsound('eng.mp3')
            # op_voice = SoundLoader.load('eng.wav')
            # print("loaded")
            # op_voice.play()

        self.scroll_to_bottom()
        self.ids.field.text = ""

    def receive(self, text):
        self.chat_logs.append(
            {
                "text": text,
                "send_by_user": False,
            }
        )

    def voice_in(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening....")
            r.pause_threshold = 0.8
            r.energy_threshold = 200
            audio = r.listen(source)

            try:
                print("Recognizing..")
                if self.flag is False:
                    toast("Please Select language!")
                    print(self.lang)
                    return
                # print(self.lang)
                query = r.recognize_google(audio, language=self.lang)
                #query = r.recognize_google(audio, language='hi-in')
                print(f"User said: {query}\n")
                print(type(query))
                print(self.lang)
                if self.lang == 'hi-in':
                    # print(f"{hindi_bot_name.text}: {out.text} \n")
                    self.chat_logs.append(
                        {"text": query, "send_by_user": True, "pos_hint": {"right": 1}}
                    )
                    rep_hindi = translator.translate(reply(query), dest='hi')
                    print(type(rep_hindi))
                    print(rep_hindi.text)
                    self.chat_logs.append(
                        {
                            "text": rep_hindi.text,
                            "send_by_user": False,
                        }
                    )
                    voice_out(rep_hindi.text, 'hi-in')

                else:
                    self.chat_logs.append(
                        {"text": query, "send_by_user": True, "pos_hint": {"right": 1}}
                    )
                    self.chat_logs.append(
                        {
                            "text": reply(query),
                            "send_by_user": False,
                        }
                    )
                    voice_out(reply(query), 'en-in')

            except Exception as e:
                # print(e)
                # print(self.lang)
                print("Say that again please...")
                return "None"

            # self.chat_logs.append(
            #     {"text": query, "send_by_user": True, "pos_hint": {"right": 1}}
            # )
            #
            # self.chat_logs.append(
            #     {
            #         "text": reply(query),
            #         "send_by_user": False,
            #     }
            # )

    # def show_user_info(self):
    #     PDialog(
    #         content=UserInfoDialogContent(
    #             title=self.user["name"],
    #             image=self.user["image"],
    #             about=self.user["about"],
    #         )
    #     ).open()

    def scroll_to_bottom(self):
        rv = self.ids.chat_rv
        box = self.ids.box
        if rv.height < box.height:
            Animation.cancel_all(rv, "scroll_y")
            Animation(scroll_y=0, t="out_quad", d=0.5).start(rv)


class UserInfoDialogContent(PBoxLayout):
    title = StringProperty()
    image = StringProperty()
    about = StringProperty()