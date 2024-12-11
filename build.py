import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import model as NN
import time
import datetime
import matplotlib.pyplot as plt
from librosa import display
from data_create import text_proc as TextProcessor
from data_create import audio_proc as AudioProcessor
from data_create import dataset as ds
from tqdm import tqdm
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

def getData(hparams, use_existing_data):
    dataset = ds.setup(hparams, use_existing_data=use_existing_data)
    train_dataset, val_dataset, test_dataset = dataset.split_sets(val=hparams.val_set_size, test=hparams.test_set_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    return train_loader, val_loader, test_loader

def getModel(hparams):
    encoder = NN.Encoder(
        vocab_size=hparams.vocab_size,
        embedding_dim=hparams.encoder_embedding_dim,
        encoder_hidden_dim=hparams.encoder_hidden_dim,
        num_layers=hparams.num_layers
    ).to(device)
    decoder = NN.Decoder(
        decoder_hidden_dim=hparams.decoder_hidden_dim,
        encoder_hidden_dim=hparams.encoder_hidden_dim,
        num_layers=hparams.num_layers,
        num_mels=hparams.n_mels
    ).to(device)
    model = NN.Seq2Seq(encoder, decoder).to(device)
    criterion = nn.L1Loss().to(device)
    optimiser = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    return model, criterion, optimiser

def collate_fn(batch):
    text_inputs, audio_targets = zip(*batch)

    # convert lists to tensors
    text_inputs = [torch.tensor(t, dtype=torch.long) for t in text_inputs]
    text_lengths = torch.tensor([len(t) for t in text_inputs], dtype=torch.long)

    # note that audio must be transposed to get shape (t, n_mels)
    audio_targets = [a.T.clone().detach().float() for a in audio_targets]
    audio_lengths = torch.tensor([a.size(0) for a in audio_targets], dtype=torch.long)

    # pad sequences
    text_inputs_padded = pad_sequence(text_inputs, batch_first=True, padding_value=0)
    audio_targets_padded = pad_sequence(audio_targets, batch_first=True, padding_value=0.0)

    return text_inputs_padded.to(device), text_lengths.to(device), audio_targets_padded.to(device), audio_lengths.to(device)

def compute_masked_loss(outputs, targets, target_lengths, criterion):
    # create mask
    max_len = targets.size(1)
    mask = torch.arange(max_len, device=device)[None, :] < target_lengths[:, None]

    # shape both to [batch_size, max_len, n_mels]
    mask = mask.unsqueeze(-1)
    outputs = outputs.transpose(1, 2)
    mask = mask.expand_as(outputs)

    # apply mask
    outputs_masked = outputs[mask]
    targets_masked = targets[mask]

    # compute loss
    loss = criterion(outputs_masked, targets_masked)
    return loss

def move_to_device(data):
    for i in range(len(data)):
        data[i] = data[i].to(device)

def train(model, train_loader, val_loader, criterion, optimiser, hparams):
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3, verbose=True)
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(hparams.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (text_inputs, text_lengths, audio_targets, audio_lengths) in tqdm(enumerate(train_loader)):
            # move data to device
            move_to_device([text_inputs, text_lengths, audio_targets, audio_lengths])
            optimiser.zero_grad()

            # run the model
            outputs = model(text_inputs, text_lengths, audio_targets, audio_lengths)

            # compute loss with masking
            loss = compute_masked_loss(outputs, audio_targets, audio_lengths, criterion)

            # prep next loop + gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        time_passed = datetime.timedelta(seconds=(time.time() - start_time))
        print(f'Epoch [{epoch+1}/{hparams.epochs}], Loss: {avg_loss:.4f}, Time: {time_passed}')

        # step the lr with a quick validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_idx, (text_val_inputs, text_val_lengths, audio_val_targets, audio_val_lengths) in enumerate(val_loader):
                # move the data to device
                move_to_device([text_val_inputs, text_val_lengths, audio_val_targets, audio_val_lengths])
                
                # run the model
                outputs_val = model(text_val_inputs, text_val_lengths, audio_val_targets, audio_val_lengths)
                
                # compute loss with masking
                loss = compute_masked_loss(outputs_val, audio_val_targets, audio_val_lengths, criterion)

                # prep next loop
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / len(val_loader)
        print(f"Validation Loss: {avg_validation_loss:.4f}")
        scheduler.step(avg_validation_loss)

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (text_test_inputs, text_test_lengths, audio_test_targets, audio_test_lengths) in enumerate(test_loader):
            # move data to device
            move_to_device([text_test_inputs, text_test_lengths, audio_test_targets, audio_test_lengths])

            # run the model
            outputs = model(text_test_inputs, text_test_lengths, audio_test_targets, audio_test_lengths)

            # compute loss with masking
            loss = compute_masked_loss(outputs, audio_test_targets, audio_test_lengths, criterion)

            # prep next loop
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss

def plotFeatures(data, data_name, save=False):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    plt.figure(figsize=(10,4))
    img = display.specshow(data, y_axis='log', x_axis='time', cmap='inferno')
    plt.title(data_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(img, format="%+2.f dB")

    if (save):
        plt.savefig(data_name+'.png')
    else:
        plt.show()
    plt.close()

def tensor_to_numpy(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if len(data.shape) == 3:
        data = data.squeeze(0)
    return data

def generate(text, model):
    textFeatures = TextProcessor.getFeatures(text)
    text_input = torch.from_numpy(textFeatures).unsqueeze(0).cuda().long()  # make tensor
    output = model.inference(text_input)
    
    # convert tensors to numpy arrays for use in 2D shape
    output = tensor_to_numpy(output)

    audio = AudioProcessor.spectrogram_to_audio(output)
    return audio, output
