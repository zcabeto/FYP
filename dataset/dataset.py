### IMPORTS ###
import pyaudio as pa
from . import extraction
from torch.utils.data import Dataset


### OVERALL DATASET ###
class TTS_Dataset(Dataset):
    def __init__(self):
        self.set = {}
        self.setlist = []
        self.mean = 0.0
        self.stdev = 1.0
    def addItem(self, item):
        self.setlist.append(item)

    def __len__(self):
        return len(self.setlist)

    def __getitem__(self, idx):
        return self.setlist[idx].getFeatures()
    
    def split_train_test(self, test_amount=0.2):
        train_set = TTS_Dataset()
        test_set = TTS_Dataset()
        index = 0
        for item in self.setlist:
            if index <= len(self.setlist)*(1-test_amount):
                train_set.addItem(item)
            else:
                test_set.addItem(item)
            index += 1
        return train_set, test_set
    
    def setStats(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev

### EACH ITEM SPLIT UP ###
class TTS_Item:
    def __init__(self, id):
        self.id = id
        self.filepath = fileFromId(id)
        
        self.textFeature = None     # numpy array
        self.audioFeature = None    # numpy array
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

def setup(limit=13100):
    extraction.extractData(dataset, limit=limit)
    return dataset