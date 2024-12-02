### IMPORTS ###
from pathlib import Path
from . import audio_proc as AudioProcess
from . import text_proc as TextProcess
import numpy as np
import time
import datetime
import sys
from math import sqrt
from . import dataset as ds

missing_ids = {}

from pathlib import Path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
audio_texts = open(root_dir+"/LJSpeech-1.1/metadata.csv", 'r')
route = root_dir + "/LJSpeech-1.1/wavs/"
route_np = root_dir + "/LJSpeech-1.1/nps/"

### AUXILLIARY FUNCTIONS ###
def makeId(validId, newNum):
    newNumStr = str(newNum)
    newNumStr = ('0'*(4-len(newNumStr)))+newNumStr
    newId = validId[:-4]+newNumStr
    return newId

def fileFromId(id):
    return route + id + '.wav'


### ITERATE THROUGH DATA ###
def nextData(destination, lineNo, n_mels):
    # get the line data
    line = audio_texts.readline()
    if not line: return False
    line_parts = line.split('|')

    # account for missing texts
    this_lineNo = int(line_parts[0][-4:])
    if this_lineNo == 1:
        lineNo = 2
    else:
        while (this_lineNo != lineNo):
            missing_ids[makeId(line_parts[0], lineNo)] = None
            lineNo += 1
        lineNo += 1

    # fix the inner data
    fileId = line_parts[0]
    text = line_parts[2].replace('\n','')
    audioFile = Path(route + fileId + '.wav')
    if audioFile.is_file():
        item = ds.TTS_Item(fileId)
        raw_audio, sr = AudioProcess.getAudio(audioFile)
        audioFeatures = AudioProcess.getFeatures(raw_audio, n_mels, sr=sr)
        textFeatures = TextProcess.getFeatures(text)
        np.save(f"{route_np}{fileId}.npy", audioFeatures)
        #item.setData(text, audioFeatures_pca, sr)
        item.setFeatures(textFeatures, audioFeatures)
        destination.addItem(item)
    else:
        missing_ids[line_parts[0]] = None

    # return to allow looping
    return np.mean(audioFeatures), np.std(audioFeatures), lineNo

def nextData_fromFile(destination, lineNo, n_mels):
    ### list through audios np file and load into items one by one instead ###
    ### list through text file and add to the appropriate item ###
    # get the line data
    line = audio_texts.readline()
    if not line: return False
    line_parts = line.split('|')

    # account for missing texts
    this_lineNo = int(line_parts[0][-4:])
    if this_lineNo == 1:
        lineNo = 2
    else:
        while (this_lineNo != lineNo):
            missing_ids[makeId(line_parts[0], lineNo)] = None
            lineNo += 1
        lineNo += 1

    # fix the inner data
    fileId = line_parts[0]
    text = line_parts[2].replace('\n','')
    audioFile = Path(route_np + fileId + '.npy')
    if audioFile.is_file():
        item = ds.TTS_Item(fileId)
        textFeatures = TextProcess.getFeatures(text)
        audioFeatures = np.load(audioFile)
        item.setFeatures(textFeatures, audioFeatures)
        destination.addItem(item)
    else:
        missing_ids[line_parts[0]] = None

    # return to allow looping
    return np.mean(audioFeatures), np.std(audioFeatures), lineNo

def extractData(destination, n_mels, use_existing_data,  limit=13100):
    count = 0
    lineNo = 1
    last_min = 0
    total_mean = 0
    total_var = 0
    start_time = time.time()
    print('Extracting data')
    while lineNo:
        if use_existing_data:
            mean, stdev, lineNo = nextData_fromFile(destination, lineNo, n_mels)
        else:
            mean, stdev, lineNo = nextData(destination, lineNo, n_mels)
        total_mean += mean
        total_var += stdev*stdev
        hit_limit, last_min = progress_tracker(count, limit, start_time, last_min)
        if hit_limit: break
        count += 1
    print('Normalising & Scaling Data')
    destination.setStats(total_mean / count, sqrt(total_var) / count)
    for item in destination.setlist:
        textFeature, audioFeature = item.getFeatures()
        normalised_audio = AudioProcess.normalise_and_scale(audioFeature, mean=mean, stdev=stdev)
        #item.audio = pca_audio
        item.setFeatures(textFeature, normalised_audio)


def progress_tracker(count, limit, start_time, last_min):
    percent = (count / limit) * 100
    time_passed = datetime.timedelta(seconds=(time.time()-start_time))
    minutes_passed = int(time_passed.total_seconds() // 60)
    if int(minutes_passed) > last_min:
        last_min = minutes_passed
        print(f"Run for {time_passed} - {percent}% complete ({count}/{limit}).")

    if count >= limit:  return True, last_min
    else:               return False, last_min