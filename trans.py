from enum import auto
import sys
print(sys.getdefaultencoding())
from googletrans import Translator
translater = Translator()

#from englisttohindi.englisttohindi import EngtoHindi

text_en = "The institute is located in forty-seven acres of green campus at Andheri (W), the fastest growing suburb of Mumbai. The campus also houses other Bhavan’s Institutions of great repute namely Bhavan’s College (the arts, commerce and science college), Sardar Patel College of Engineering, S.P. Jain Institute of Management and Research, a management institute. It also has an Auditorium, Bhavan's Lake and a common cricket ground for the campus institutes. SPIT, for its own, does not have any hostel facilities for students. In addition to this, SPIT also owns its own Technology Incubation Center for Startups named SP-TBI"
text_hi = "आप कैसे हो?"
#print(text_hi)
print(translater.detect(text_hi))
#print(EngtoHindi(text_en).convert)
out = translater.translate(text_en, dest='hi')
print(out.text)