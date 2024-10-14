### IMPORTS ###
import pyaudio as pa
import extraction


### OVERALL DATASET ###
class TTS_Dataset:
    def __init__(self):
        self.set = {}
    def addItem(self, id, text):
        item = TTS_Item(fileFromId(id), text)
        self.set[id] = item
        return item
    def items(self):
        return self.set.items()

### EACH ITEM SPLIT UP ###
class TTS_Item:
    def __init__(self, audioFile, text):
        self.file = audioFile
        self.text = text
        self.data = None
        self.sr = None

    def setAudio(self, data, sr):
        self.data = data
        self.sr = sr

### AUXILLIARY FUNCTIONS ###
def fileFromId(id):
    return route + id + '.wav'

### RUNNABLE ###
missing_ids = {}
route = './LJSpeech-1.1/wavs/'
dataset = TTS_Dataset()
audio_texts = open('./LJSpeech-1.1/metadata.csv','r')

def setup():
    extraction.extractData(dataset, 20)
    return dataset