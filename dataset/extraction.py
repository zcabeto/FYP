### IMPORTS ###
from pathlib import Path
import audio_proc as AudioProcess
import text_proc as TextProcess


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
    text = TextProcess.cleanText(line_parts[2].replace('\n',''))
    audioFile = Path(route + fileId + '.wav')
    if audioFile.is_file():
        item = destination.addItem(fileId, text)
        AudioProcess.extractAudioData(item)
    else:
        missing_ids[line_parts[0]] = None
    
    # return to allow looping
    return lineNo

def extractData(destination, limit=None):
    total_count = 0
    lineNo = 1
    while lineNo:
        lineNo=nextData(destination, lineNo)
        total_count += 1
        if limit: 
            if total_count > limit: break