import sounddevice as sd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_create import dataset as ds
from data_create import text_proc as TextProcessor
import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
from build import train, test, collate_fn, getData, getModel
import model as NN
from data_create import audio_proc as AudioProcessor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 70         # = len(ARPAbet)
embedding_dim = 4900    # likely 50-100 layers for <100,000 sentences
hidden_dim = 140        # less than embedding_dim to combine & not overfit
num_layers = 1          # start at 2, increase for more quality
num_mels = 256          # 256 best quality, 80 smallest size. Proper run ~1000
num_epochs = 5
learning_rate = 0.01
batch_size = 8          # how many to process at once, probs small to ensure matchup

print('Initialising Data')
train_set, val_set, test_set = getData(num_mels=num_mels, batch_size=batch_size)

print('Initialising Model')
model, criterion, optimiser = getModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_mels=num_mels,
    learning_rate=learning_rate
)

print("Training Model")
train(model, device, train_set, val_set, criterion, optimiser, num_epochs)

'''print("Testing Model")
test(model, device, test_set, criterion, optimiser)'''


