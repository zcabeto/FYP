### IMPORTS ###
import sounddevice as sd
import pyaudio as pa
import soundfile as sf
import librosa as lb
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter



sr = 22050

### OVERALL MANAGER ###
def getAudio(filepath):
    raw_audio, sr = lb.load(filepath, sr=None)

    audio = trim_edges(raw_audio)
    audio = spectral_gate(audio)
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

def spectral_gate(y, n_fft=2048, hop_length=512, win_length=None):
    # Short-time Fourier transform
    stft_matrix = lb.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitude, phase = np.abs(stft_matrix), np.angle(stft_matrix)

    # Estimate the noise level by averaging the silent frames
    noise_mag = np.mean(magnitude[:, :10], axis=1)  # adjust based on your data
    spectral_gate_threshold = noise_mag * 1.5  # adjust the threshold based on your needs

    # Suppress noise by spectral gating
    magnitude[magnitude < spectral_gate_threshold[:, None]] = 0

    # Reconstruct the signal
    filtered_stft = magnitude * np.exp(1j * phase)
    y_reduced_noise = lb.istft(filtered_stft, hop_length=hop_length, win_length=win_length)
    return y_reduced_noise

def reduce_noise_with_pca(spectrogram, variance_threshold=0.9):
    spectrogram = spectrogram.T
    # Perform PCA to determine number of components to keep
    pca = PCA()
    pca.fit(spectrogram)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
    # Apply PCA with the desired number of components
    pca = PCA(n_components=n_components)
    spectrogram_transformed = pca.fit_transform(spectrogram)

    # Reconstruct the spectrogram
    spectrogram_reconstructed = pca.inverse_transform(spectrogram_transformed)
    spectrogram_reconstructed = spectrogram_reconstructed.T

    # deviate useful data from non-useful data
    for r in range(spectrogram_reconstructed.shape[0]):
        for c in range(spectrogram_reconstructed.shape[1]):
            if spectrogram_reconstructed[r,c] < -5:
                spectrogram_reconstructed[r,c] *= 1.5
            if spectrogram_reconstructed[r,c] < -15:
                spectrogram_reconstructed[r,c] = -15

    return spectrogram_reconstructed



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
    return reduce_noise_with_pca(spectrogram_normalised)
    #return spectrogram_normalised

### REVERSAL TO PLAY ###
def play_mel_spectrogram(spectrogram, mean, stdev, sr=22050):
    spectrogram = denormalise(spectrogram, mean, stdev)
    spectrogram = mel_to_linear(spectrogram)
    audio = reconstruct_waveform(spectrogram)
    playGeneric(audio, sr=sr)

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

def playGeneric(audio, sr=22050):
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)
    p = pa.PyAudio()
    try:
        stream = p.open(format=pa.paInt16, channels=1, rate=sr, output=True)
        stream.write(audio.tobytes())
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()
