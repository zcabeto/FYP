import sys
import torch
from scipy.io import wavfile
import soundfile as sf
import numpy as np
import os
import argparse
import warnings
import librosa as lb
from rich.progress import track

def add_noise(audio, noise_file, level=1):
    '''
    Add noise to an audio file, chosen by file
    files are from https://www.epidemicsound.com/sound-effects/
    '''
    if not os.path.exists(noise_file):
        return audio    # no noise listed
    noise, _ = sf.read(noise_file)
    
    # resize noise & audio to same shape
    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1)
    audio = audio.flatten()
    if len(noise) > len(audio):
        start_idx = np.random.randint(0, len(noise) - len(audio))
        noise = noise[start_idx : start_idx+len(audio)]
    else:
        noise = np.pad(noise, (0, len(audio) - len(noise)), 'wrap')
    
    # put together & renormalise
    noise = noise / np.max(np.abs(noise))
    audio = (audio + (noise * level)) / 2
    return audio

def change_quality(audio_numpy, quality):
    '''
    Change the quality of an audio file by a given factor
        quality > 1 increases the quality
        quality < 1 decreases the quality
    '''
    assert quality > 0, "Quality must be greater than 0"
    if quality <= 0.1:
        target_sr = 4000
    elif quality <= 0.3:
        target_sr = 8000
    elif quality <= 0.7:
        target_sr = 16000
    elif quality <= 1.0:
        target_sr = 22050  # default
    elif quality <= 1.5:
        target_sr = 32000
    else:
        target_sr = 44100

    # resample to target sample rate if needed
    if target_sr != 22050:
        audio_numpy = audio_numpy.flatten()
        audio_numpy = lb.resample(audio_numpy, orig_sr=22050, target_sr=target_sr)
    return audio_numpy, target_sr

warnings.filterwarnings("ignore", message=".*pytorch_quantization module not found.*")
class suppress_stdout_stderr(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

print("Loading WaveGlow Model...")
with suppress_stdout_stderr():
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32')

waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--spectrogram_dir', type=str, default="spectrogram_generation/spectrograms")
    parser.add_argument('-a', '--audio_dir', type=str, default="audio_generation/audios")
    parser.add_argument('-n', '--noise_dir', type=str, default="audio_generation/background_noise")
    args = parser.parse_args()
    for filename in track(os.listdir(args.spectrogram_dir), "Generating Audios"):
        file_path = os.path.join(args.spectrogram_dir, filename)
        mel_spectrogram = np.load(file_path)
        mel_spectrogram = torch.from_numpy(mel_spectrogram).cuda().float()
        mel_spectrogram = mel_spectrogram.unsqueeze(0)

        _, id, noise_file, level, quality = filename[:-4].split('-')    # remove .npy
        with torch.no_grad():
            audio = waveglow.infer(mel_spectrogram, sigma=0.666)
        audio_numpy = audio[0].data.cpu().numpy()

        # add background noise and change quality before saving
        audio_numpy = add_noise(audio_numpy, f"{args.noise_dir}/{noise_file}.wav", level=float(level))
        audio_numpy, sr = change_quality(audio_numpy, quality=float(quality))
        wavfile.write(f"{args.audio_dir}/audio-{id}.wav", sr, (audio_numpy * 32767).astype(np.int16))
