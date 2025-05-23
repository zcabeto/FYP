### IMPORTS ###
import librosa as lb
import numpy as np
import torch
from .stft import STFT
from scipy.io import wavfile


### SETUP & RUN ###
mag_stft = None
mel_filter = None
hparams = None
def setup_stft(params):
    global mag_stft
    global mel_filter
    global hparams
    if (mag_stft is not None) and (mel_filter is not None) and (hparams is not None):
        return
    hparams = params
    mag_stft = STFT(filter_length=hparams.filter_length, hop_length=hparams.hop_length, win_length=hparams.win_length)
    mel_filter = lb.filters.mel(
            sr=hparams.sr, 
            n_fft=hparams.filter_length, 
            n_mels=hparams.n_mels, 
            fmin=hparams.fmin, 
            fmax=hparams.fmax)
    mel_filter = torch.from_numpy(mel_filter).float()

def audioFeatures(audio_file):
    if mag_stft is None or mel_filter is None:
        raise RuntimeError("STFT not initialized")
    audio = getAudio(audio_file)
    mel_spectrogram = audio_to_spectrogram(audio)
    return mel_spectrogram

### AUDIO CLEANING ###
def getAudio(filepath):
    sr, raw_audio = wavfile.read(filepath)
    if sr != hparams.sr:
        raise ValueError(f"Sampling Rate wrong - got {sr}")
    #audio = trim_edges(raw_audio)
    #audio = cut_pauses(audio)
    audio_tensor = convert_to_tensor(raw_audio)
    return audio_tensor

def trim_edges(audio, top_db=40):
    audio, i = lb.effects.trim(audio, top_db=top_db)
    return audio

def cut_pauses(audio, top_db=40, hop_length=1000):
    audio_parts_ranges = lb.effects.split(audio,top_db=top_db, hop_length=hop_length)
    segments = []
    for start, end in audio_parts_ranges:
        segment = audio[start:end]
        segments.append(segment)
    restitched = np.concatenate(segments)
    return restitched

def convert_to_tensor(audio):
    return torch.FloatTensor(audio.astype(np.float32))

### FEATURE GENERATION ###
def audio_to_spectrogram(audio):
    audio_normalised = audio / hparams.max_wav    # normalise audio
    audio_normalised = audio_normalised.unsqueeze(0)    # add batch dimension

    # main computation of mel_spectrogram
    magnitude_spectrogram,_ = mag_stft.transform(audio_normalised)
    mel_spectrogram = torch.matmul(mel_filter, magnitude_spectrogram.data)
    mel_spectrogram = torch.clamp(mel_spectrogram, min=1e-5)    # prevent log(0) (previously 1e-8)
    mel_spectrogram = torch.log(mel_spectrogram)                # linear to decibel

    mel_spectrogram = torch.squeeze(mel_spectrogram, 0) # remove batch dimension
    return mel_spectrogram

### REVERSAL TO PLAY ###
def spectrogram_to_audio(mel_spectrogram):
    mel_spectrogram = np.exp(mel_spectrogram)           # decibel to linear
    spectrogram = mel_to_magnitude(mel_spectrogram)
    spectrogram = spectrogram * hparams.max_wav   # denormalise audio
    audio = reconstruct_waveform(spectrogram)
    return audio

def reconstruct_waveform(mag_spectrogram, n_iter=1000, window='hann'):
    waveform = lb.griffinlim(
        mag_spectrogram,
        n_iter=n_iter,
        hop_length=hparams.hop_length,
        win_length=hparams.win_length,
        window=window,
        n_fft=hparams.filter_length
    )
    return waveform

def mel_to_magnitude(mel_spectrogram):
    mag_spectrogram = lb.feature.inverse.mel_to_stft(
        M=mel_spectrogram,
        sr=hparams.sr,
        n_fft=hparams.filter_length,
        power=2.0
    )
    return mag_spectrogram
