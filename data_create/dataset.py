### IMPORTS ###
from . import extraction
from torch.utils.data import Dataset

### DATASET CLASSES ###
class TTS_Dataset(Dataset):
    def __init__(self):
        self.setlist = []

    def addItem(self, item):
        self.setlist.append(item)

    def __len__(self):
        return len(self.setlist)

    def __getitem__(self, idx):
        return self.setlist[idx].getFeatures()

    def split_sets(self, val, test):
        train_set = TTS_Dataset()
        val_set = TTS_Dataset()
        test_set = TTS_Dataset()
        index = 0
        for item in self.setlist:
            if index <= len(self.setlist)*(1-test):
                if index > len(self.setlist)*(1-test-val):
                    val_set.addItem(item)
                else:
                    train_set.addItem(item)
            else:
                test_set.addItem(item)
            index += 1
        return train_set, val_set, test_set

class TTS_Item:
    def __init__(self, id):
        self.id = id
        self.textFeature = None
        self.audioFeature = None

    def setFeatures(self, textFeature, audioFeature):
        self.textFeature = textFeature
        self.audioFeature = audioFeature

    def getFeatures(self):
        return self.textFeature, self.audioFeature

### SETUP DATASET ###
dataset = TTS_Dataset()
def setup(hparams, use_existing_data=False):
    extraction.extractData(dataset, hparams, use_existing_data=use_existing_data)
    return dataset