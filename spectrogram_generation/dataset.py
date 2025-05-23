### IMPORTS ###
from torch.utils.data import Dataset
from .text_proc import textFeatures
from .audio_proc import audioFeatures
import random


### DATASET CLASSES ###
class TTS_Dataset(Dataset):
    def __init__(self, hparams, setlist):
        self.setlist = setlist
        random.seed(hparams.seed)
        random.shuffle(self.setlist)

    def __len__(self):
        return len(self.setlist)

    def __getitem__(self, idx):
        return self.setlist[idx].getFeatures()

class TTS_Item:
    def __init__(self, audio_filename, text_raw):
        self.audio_filename = audio_filename
        self.text_raw = text_raw
    
    def getFeatures(self):
        return textFeatures(self.text_raw), audioFeatures(self.audio_filename)

def create_dataset(hparams):
    with open(hparams.metadata_file, encoding='utf-8') as f:
        train_filelist = []
        test_filelist = []
        val_filelist = []
        id = 0
        for line in f:
            line_parts = line.strip().split('|')
            if id < hparams.training_n:
                train_filelist.append(TTS_Item(line_parts[0], line_parts[1]))
            elif id < hparams.training_n + hparams.testing_n:
                test_filelist.append(TTS_Item(line_parts[0], line_parts[1]))
            elif id < hparams.n:
                val_filelist.append(TTS_Item(line_parts[0], line_parts[1]))
            id += 1
    train_set = TTS_Dataset(hparams, train_filelist)
    val_set = TTS_Dataset(hparams, val_filelist)
    test_set = TTS_Dataset(hparams, test_filelist)
    return train_set, val_set, test_set
