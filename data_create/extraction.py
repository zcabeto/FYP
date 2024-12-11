from pathlib import Path
from . import audio_proc as AudioProcess
from . import text_proc as TextProcess
import numpy as np
from . import dataset as ds
from tqdm import tqdm
import torch
from pathlib import Path


root_dir = str(Path(__file__).resolve().parent.parent.parent)
audio_texts = open(root_dir+"/LJSpeech-1.1/metadata.csv", 'r')
route = root_dir + "/LJSpeech-1.1/wavs/"
route_np = root_dir + "/LJSpeech-1.1/nps/"

def extractData(destination, hparams, use_existing_data=False):
    count = 0
    AudioProcess.setup_stft(hparams)
    for i in tqdm(range(hparams.n)):
        nextData(destination, use_existing_data)
        count += 1
    audio_texts.close()

def nextData(destination, use_existing_data):
    # get the line data
    line = audio_texts.readline()
    line_parts = line.split('|')
    fileId = line_parts[0]
    text = line_parts[2].replace('\n','')

    # get the features of the data
    item = get_features_old(fileId, text) if use_existing_data else get_features_new(fileId, text)
    destination.addItem(item)

def get_features_new(fileId, text):
    audioFile = Path(route + fileId + '.wav')
    item = None
    if audioFile.is_file():
        item = ds.TTS_Item(fileId)
        audioFeatures = AudioProcess.getFeatures(audioFile)
        textFeatures = TextProcess.getFeatures(text)
        np.save(f"{route_np}{fileId}.npy", audioFeatures)
        item.setFeatures(textFeatures, audioFeatures)
    else:
        print(f"File {fileId} not found")
    return item

def get_features_old(fileId, text):
    audioFile = Path(route_np + fileId + '.npy')
    item = None
    if audioFile.is_file():
        item = ds.TTS_Item(fileId)
        textFeatures = TextProcess.getFeatures(text)
        audioFeatures = torch.from_numpy(np.load(audioFile))
        item.setFeatures(textFeatures, audioFeatures)
    else:
        print(f"File {fileId} not found")
    return item