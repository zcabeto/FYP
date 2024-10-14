### IMPORTS ###
import sounddevice as sd
import pyaudio as pa
import soundfile as sf
import librosa as lb
import numpy as np


### EXTRACT AUDIO DATA ###
def format_audio(data):
    return np.int16(data / np.max(np.abs(data)) * 32767)
    
def extractAudioData(item):
    unformatted_data, samplerate = lb.load(item.file, sr=None)
    unformatted_data, index = lb.effects.trim(unformatted_data, top_db=40)
    unformatted_data, samplerate = processAudio(unformatted_data, samplerate)
    item.setAudio(format_audio(unformatted_data), samplerate)

### LARGE MANAGER ###
def processAudio(audio, sr):
    return audio, sr
    #data = lb.effects.percussive(lb.effects.pitch_shift(data, sr=samplerate, n_steps=4))

### INDIVIDUAL EFFECTS ###



### unused ###
def cut_at_pauses(text, audio):
    #text = re.sub('["~@£$%^*]','',re.sub("'",'',re.sub('-',' ',text))).lower()
    #text_parts = re.split('[,;:]',text)
    text_parts = text.split(',')
    if text_parts[-1] == '': text_parts = text_parts[:-1]
    if text_parts[0] == '': text_parts = text_parts[1:]
    #print(len(text_parts))
    if (len(text_parts) == 1): return [text], [audio]
    audio_parts_ranges = []
    top_db = 20
    max_db = 40
    while top_db <= max_db:
        audio_parts_ranges = lb.effects.split(audio,top_db=top_db, hop_length=4000)
        #print(top_db, len(audio_parts_ranges))
        if len(text_parts) == len(audio_parts_ranges):
            break
        top_db += 5
    if len(text_parts) != len(audio_parts_ranges): return [text], [audio]
    audio_parts = []
    for i in range(len(audio_parts_ranges)):
        audio_parts.append(audio[audio_parts_ranges[i,0] : audio_parts_ranges[i,1]])
        text_parts[i] = text_parts[i].strip()
    print(top_db)
    return text_parts, audio_parts   