import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_create import dataset as ds
from data_create import text_proc as TextProcessor
import librosa as lb
import numpy as np
from build import train, test, collate_fn, getData, getModel, plotFeatures
import model as NN
from data_create import audio_proc as AudioProcessor

vocab_size = 70         # = len(ARPAbet)
embedding_dim = 4900    # likely 50-100 layers for <100,000 sentences
hidden_dim = 140        # less than embedding_dim to combine & not overfit
num_layers = 2          # start at 2, increase for more quality
num_mels = 256          # 256 best quality, 80 smallest size. Proper run ~1000
num_epochs = 11
learning_rate = 0.01
batch_size = 32         # how many to process at once, probs small to ensure matchup
use_existing_data = True
n = 13000                 # number of datapoints to run

if __name__ == '__main__':
    print('Initialising Data')
    train_set, val_set, test_set = getData(num_mels=num_mels, batch_size=batch_size, n=n, use_existing_data=use_existing_data)

    print('Initialising Model')
    model, criterion, optimiser, device = getModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_mels=num_mels,
        learning_rate=learning_rate
    )

    print("Training Model")
    train(model, train_set, val_set, criterion, optimiser, num_epochs)

    print("Testing Model")
    test(model, test_set, criterion, optimiser)

'''
def spectral_subtraction(raw_spectrogram, noise_spectrogram, alpha=2.0, beta=1.0):
    """
    Perform spectral subtraction to denoise a spectrogram.

    Parameters:
    - raw_spectrogram: np.ndarray, shape (freq_bins, time_frames)
    - noise_spectrogram: np.ndarray, shape (freq_bins, noise_time_frames)
    - alpha: float, over-subtraction factor
    - beta: float, flooring parameter to prevent negative values

    Returns:
    - denoised_spectrogram: np.ndarray, same shape as raw_spectrogram
    """
    # Compute the average noise spectrum across time frames
    average_noise_spectrum = np.mean(noise_spectrogram, axis=1, keepdims=True)
    noise_spectrogram_expanded = np.repeat(average_noise_spectrum, raw_spectrogram.shape[1], axis=1)
    plotFeatures(noise_spectrogram_expanded, 'noise', save=True)

    # Perform spectral subtraction
    denoised = raw_spectrogram + alpha * noise_spectrogram_expanded
    denoised += np.full(denoised.shape, 15)
    plotFeatures(denoised, 'denoised0', save=True)
    denoised1 = np.minimum(denoised, raw_spectrogram)
    plotFeatures(denoised1, 'denoised1', save=True)
    denoised2 = np.maximum(denoised, raw_spectrogram) - np.full(denoised.shape, 5)
    plotFeatures(denoised2, 'denoised2', save=True)
    denoised3 = np.copy(raw_spectrogram)
    for r in range(denoised3.shape[0]):
        for c in range(denoised3.shape[1]):
            if denoised3[r,c] < -5:
                denoised3[r,c] *= 1.5
            if denoised3[r,c] < -15:
                denoised3[r,c] = -15
    plotFeatures(denoised3, 'denoised3', save=True)
    AudioProcessor.play_mel_spectrogram(denoised3, mean=dataset.mean, stdev=dataset.stdev)
    #return denoised, noise_spectrogram_expanded


for i in range(3):
    textFeatures, reconstructed_spectrogram, train_set.__getitem__(i)
    text_input = torch.tensor(textFeatures, dtype=torch.long).unsqueeze(0).to(device)
    generated_spectrogram = model.generate(text_input)
    plotFeatures(generated_spectrogram, 'generated'+str(i), save=True)
    plotFeatures(reconstructed_spectrogram, 'reconstructed'+str(i), save=True)



print(text)
input()
print('original (reconstructed)')
AudioProcessor.play_mel_spectrogram(reconstructed_spectrogram, mean=dataset.mean, stdev=dataset.stdev)
#input()
#print('generated noise')
#AudioProcessor.play_mel_spectrogram(generated_spectrogram, mean=dataset.mean, stdev=dataset.stdev)
input()
print('denoised audio')
AudioProcessor.play_mel_spectrogram(denoised_spectrogram, mean=dataset.mean, stdev=dataset.stdev)'''