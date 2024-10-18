### IMPORTS ###
import sounddevice as sd
import pyaudio as pa
import soundfile as sf
import librosa as lb
import numpy as np


### OVERALL MANAGER ###
def getAudio(filepath):
    raw_audio, sr = lb.load(filepath, sr=None)
    processed_audio = processAudio(raw_audio)
    playable_audio = format_audio(processed_audio)
    return processed_audio, playable_audio, sr

### GENERAL AUDIO CLEANING ###
def processAudio(audio):
    audio = trim_edges(audio)
    audio = cut_pauses(audio)
    return audio

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

### PREPARE TO PLAY ###
def format_audio(data):
    return np.int16(data / np.max(np.abs(data)) * 32767)

### PREPARE FEATURES ###
def getFeatures(audio):
    spectrogram = make_spectrogram(audio)
    spectrogram = normalise_and_scale(spectrogram)
    return spectrogram

def make_spectrogram(audio, mel=True):
    spectrogram = lb.stft(audio)
    # convert to mel spectrogram
    return spectrogram

### NORMALISATION & SCALING ###
def normalise_and_scale(audio):
    return audio

def amplitude_to_db(audio):
    return audio 