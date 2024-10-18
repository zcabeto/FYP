### IMPORTS ###
from pathlib import Path
import audio_proc as AudioProcess
import text_proc as TextProcess
import time
import datetime


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
def nextData(destination, lineNo):
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
        item = destination.addItem(fileId)
        raw_audio, playable_audio, sr = AudioProcess.getAudio(audioFile)
        audioFeatures = AudioProcess.getFeatures(raw_audio, sr)
        textFeatures = TextProcess.getFeatures(text)
        item.setData(text, playable_audio, sr)
        item.setFeatures(textFeatures, audioFeatures)
    else:
        missing_ids[line_parts[0]] = None
    
    # return to allow looping
    return lineNo

def extractData(destination, limit=13100):
    count = 0
    lineNo = 1
    start_time = time.time()
    while lineNo:
        lineNo=nextData(destination, lineNo)
        if progress_tracker(count, limit, start_time): break
        count += 1

def progress_tracker(count, limit, start_time):
    now_fract = (count / limit)
    last_fract = ((count-1) / limit)
    if int(now_fract * 10) > int(last_fract * 10):
        message = ("Extraction {percent}% complete ({count}/{limit}). Run for {time}.").format(percent=int(now_fract*10)*10, count=count, limit=limit, time=datetime.timedelta(seconds=(time.time()-start_time)))
        print(message)
    if count >= limit: return True
    else: return False
