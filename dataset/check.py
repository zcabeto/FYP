### IMPORTS ###
import sounddevice as sd
import pyaudio as pa
import soundfile as sf
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
import random
import dataset as ds

### PLAY AUDIO FILES ###
def checkAudioRandom():
    id, item = random.sample(list(dataset.items()), 5)[0]
    print(id)
    print(item.text)
    playGeneric(item.data, item.sr)
    return item.text, item.data, item.sr

def checkAudioSpecific(id):
    if dataset.set[id]:
        print(id)
        print(dataset.set[id].text)
        playGeneric(dataset.set[id].data, dataset.set[id].sr)
        return dataset.set[id].text, dataset.set[id].data, dataset.set[id].sr

def playGeneric(data, sr):
    p = pa.PyAudio()
    try:
        stream = p.open(format=pa.paInt16, channels=1, rate=sr, output=True)
        stream.write(data.tobytes())
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()


### PLOT AUDIO DATA ###
def plot(data, sr, data_name, save=False):  # note fft (dense sampling FT), hopsize, windowsize stuff for later
    spectrogram = np.abs(lb.stft(data))                 # get short-term fourier transform for amplitude
    spectrogram_db = lb.amplitude_to_db(spectrogram)    # convert to decibels

    plt.figure(figsize=(10,4))
    img = lb.display.specshow(spectrogram_db, y_axis='log', x_axis='time', sr=sr, cmap='inferno')
    plt.title(data_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.yticks(np.array([numbers]), np.array([labels])) to change the values on the axis
    plt.colorbar(img, format="%+2.f dB")
    
    if (save):
        plt.savefig(data_name+'.png')
    else:
        plt.show()

dataset = ds.setup()
checkAudioRandom()