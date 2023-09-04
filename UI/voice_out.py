from gtts import gTTS
from kivy.core.audio import SoundLoader
import os


def voice_out(text, lang):
    if lang == 'hi-in':
        os.remove('out.mp3')
        op_voice = gTTS(text, slow=False, lang='hi')
        op_voice.save('out.mp3')
        op_voice = SoundLoader.load('out.mp3')
        play_voice(op_voice)
    else:
        op_voice = gTTS(text, slow=False, lang='en')
        op_voice.save('out.mp3')
        op_voice = SoundLoader.load('out.mp3')
        print(op_voice.state)
        play_voice(op_voice)


def play_voice(op_voice):
    # oss_audio_device.close()
    op_voice.play()
