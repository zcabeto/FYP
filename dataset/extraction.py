### IMPORTS ###
from pathlib import Path
from . import audio_proc as AudioProcess
from . import text_proc as TextProcess
import numpy as np
import time
import datetime
from . import dataset as ds

missing_ids = {}
route = './LJSpeech-1.1/wavs/'
audio_texts = open('./LJSpeech-1.1/metadata.csv','r')


### AUXILLIARY FUNCTIONS ###
def makeId(validId, newNum):
    newNumStr = str(newNum)
    newNumStr = ('0'*(4-len(newNumStr)))+newNumStr
    newId = validId[:-4]+newNumStr
    return newId

def fileFromId(id):
    return route + id + '.wav'


### ITERATE THROUGH DATA ###
def nextData(destination, lineNo, total_audio):
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
        audioFeatures = AudioProcess.getFeatures(raw_audio, sr)
        total_audio = np.concatenate((total_audio, audioFeatures.flatten()))
        textFeatures = TextProcess.getFeatures(text)
        item.setData(text, raw_audio, sr)
        item.setFeatures(textFeatures, audioFeatures)
        destination.addItem(item)
    else:
        missing_ids[line_parts[0]] = None
    
    # return to allow looping
    return lineNo, total_audio

def extractData(destination, limit=13100):
    last_min = 0
    count = 0
    lineNo = 1
    total_audio = np.array([])
    start_time = time.time()
    print('Extracting data')
    while lineNo:
        lineNo, total_audio = nextData(destination, lineNo, total_audio)
        if progress_tracker(count, limit, start_time, last_min): break
        count += 1
    print('Normalising & Scaling Data')
    mean = np.mean(total_audio)
    stdev = np.std(total_audio)
    destination.setStats(mean, stdev)
    for item in destination.setlist:
        textFeature, audioFeature = item.getFeatures()
        normalised_audio = AudioProcess.normalise_and_scale(audioFeature, mean=mean, stdev=stdev)
        item.setFeatures(textFeature, normalised_audio)
    

def progress_tracker(count, limit, start_time, last_min):
    percent = (count / limit) * 100
    time_passed = datetime.timedelta(seconds=(time.time()-start_time))
    minutes_passed = int(time_passed.total_seconds() // 60)
    if int(minutes_passed) > last_min:
        last_min += time_passed.min
        print(f"Extraction {percent}% complete ({count}/{limit}). Run for {time_passed}.")
    if count >= limit: return True
    else: return False
