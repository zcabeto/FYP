### IMPORTS ###
import pyaudio as pa
import extraction


### OVERALL DATASET ###
class TTS_Dataset:
    def __init__(self):
        self.set = {}
    def addItem(self, id):
        item = TTS_Item(id)
        self.set[id] = item
        return item
    def items(self):
        return self.set.items()

### EACH ITEM SPLIT UP ###
class TTS_Item:
    def __init__(self, id):
        self.id = id
        self.filepath = fileFromId(id)
        
        self.textFeature = None
        self.audioFeature = None
        self.text = None
        self.audio = None
        self.sr = None

    def setFeatures(self, textFeature, audioFeature):
        self.textFeature = textFeature
        self.audioFeature = audioFeature

    def getFeatures(self):
        return self.textFeature, self.audioFeature

    def setData(self, text, audio, sr):
        self.text = text
        self.audio = audio
        self.sr = sr
    
    def getData(self):
        return self.text, self.audio, self.sr
    
### AUXILLIARY FUNCTIONS ###
def fileFromId(id):
    return route + id + '.wav'

### RUNNABLE ###
missing_ids = {}
route = './LJSpeech-1.1/wavs/'
dataset = TTS_Dataset()
audio_texts = open('./LJSpeech-1.1/metadata.csv','r')

def setup():
    extraction.extractData(dataset, 100)
    return dataset