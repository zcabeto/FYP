### IMPORTS ###
import librosa as lb
import numpy as np



sr = 22050

### OVERALL MANAGER ###
def getAudio(filepath):
    raw_audio, sr = lb.load(filepath, sr=None)

    audio = trim_edges(raw_audio)
    audio = cut_pauses(audio)
    return audio, sr

### GENERAL AUDIO CLEANING ###
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



### PREPARE FEATURES ###
def getFeatures(audio, n_mels, sr=22050, mel=True, n_fft=1024, hop_length=256):
    if mel:
        spectrogram = lb.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
    else:
        spectrogram = lb.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(spectrogram) ** 2
    return spectrogram

def normalise_and_scale(spectrogram, mean, stdev):
    spectrogram = lb.power_to_db(spectrogram, ref=1.0)
    spectrogram_normalised = (spectrogram - mean) / stdev
    #return spectrogram_normalised, reduce_noise_with_pca(spectrogram_normalised)
    return spectrogram_normalised

### REVERSAL TO PLAY ###
def spectrogram_to_audio(spectrogram, mean, stdev, sr=22050):
    spectrogram = denormalise(spectrogram, mean, stdev)
    spectrogram = mel_to_linear(spectrogram)
    audio = reconstruct_waveform(spectrogram)
    return audio

def reconstruct_waveform(linear_spectrogram, n_fft=1024, hop_length=256, win_length=None, window='hann', n_iter=200):
    waveform = lb.griffinlim(
        linear_spectrogram,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        n_fft=n_fft
    )
    return waveform

def mel_to_linear(mel_spectrogram, sr=22050, n_fft=1024):
    linear_spectrogram = lb.feature.inverse.mel_to_stft(
        M=mel_spectrogram,
        sr=sr,
        n_fft=n_fft,
        power=2.0
    )
    return linear_spectrogram


def denormalise(spectrogram, mean, stdev):
    spectrogram = (spectrogram * stdev) + mean
    spectrogram = lb.db_to_power(spectrogram, ref=1.0)
    return spectrogram

