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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import build
import model as NN


# hollow numbers for now
vocab_size = 100
embedding_dim = 256
hidden_dim = 512
num_layers = 2
num_mels = 80
num_epochs = 1
learning_rate = 0.001

# Create the datasets
print('Creating Datasets')
dataset = ds.setup(limit=10)
train_dataset, test_dataset = dataset.split_train_test()
train_loader = DataLoader(
    train_dataset,
    batch_size=train_dataset.__len__(),
    shuffle=True,
    collate_fn=build.collate_fn
)
test_loader = DataLoader(
    test_dataset,
    batch_size=test_dataset.__len__(),
    shuffle=False,
    collate_fn=build.collate_fn
)

# Instantiate the model
encoder = NN.Encoder(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    encoder_hidden_dim=hidden_dim,
    num_layers=num_layers
)
decoder = NN.Decoder(
    decoder_hidden_dim=hidden_dim,
    encoder_hidden_dim=hidden_dim,
    num_layers=num_layers,
    num_mels=num_mels
)
model = NN.Seq2Seq(encoder, decoder).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Beginning Training")
#train(model, device, train_loader, criterion, optimizer)
print("Training complete.")
#test(model, device, test_loader, criterion, optimizer)
#print("Testing complete. Now to have a go")

