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
import numpy as np
from data_create import text_proc as TextProcessor
from data_create import dataset as ds

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')

def getData(num_mels, batch_size, n, use_existing_data):
    dataset = ds.setup(num_mels, limit=n, use_existing_data=use_existing_data)
    train_dataset, val_dataset, test_dataset = dataset.split_sets(val=10)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    return train_loader, val_loader, test_loader

def getModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_mels, learning_rate):
    encoder = NN.Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)
    decoder = NN.Decoder(
        decoder_hidden_dim=hidden_dim,
        encoder_hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_mels=num_mels
    ).to(device)
    model = NN.Seq2Seq(encoder, decoder).to(device)
    criterion = nn.L1Loss().to(device)
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimiser, device

def collate_fn(batch):
    text_inputs, audio_targets = zip(*batch)

    # Convert lists to tensors
    text_inputs = [torch.tensor(t, dtype=torch.long) for t in text_inputs]
    text_lengths = torch.tensor([len(t) for t in text_inputs], dtype=torch.long)

    # note that audio must be transposed to get shape (t, n_mels)
    audio_targets = [torch.tensor(a.T, dtype=torch.float) for a in audio_targets]
    audio_lengths = torch.tensor([a.size(0) for a in audio_targets], dtype=torch.long)

    # Pad sequences
    text_inputs_padded = pad_sequence(text_inputs, batch_first=True, padding_value=0)
    audio_targets_padded = pad_sequence(audio_targets, batch_first=True, padding_value=0.0)

    return text_inputs_padded.to(device), text_lengths.to(device), audio_targets_padded.to(device), audio_lengths.to(device)


def train(model, train_loader, val_loader, criterion, optimiser, num_epochs):
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=3, verbose=True)
    torch.backends.cudnn.benchmark = True
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Optimiser device: {optimiser.param_groups[0]['params'][0].device}")
    print(f"Criterion device: {criterion.weight.device if hasattr(criterion, 'weight') else 'N/A'}")
    textFeatures = TextProcessor.getFeatures("hello world")
    text_input = torch.tensor(textFeatures, dtype=torch.long).unsqueeze(0).to(device)
    for epoch in range(num_epochs):
        generated_spectrogram = model.generate(text_input)
        plotFeatures(generated_spectrogram, '../../../../../ug/2021/etomlin/imgs/generated_'+str(epoch), save=True)
        '''print(f"Epoch {epoch} parameters...")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.data.mean().item():.4f}, std={param.data.std().item():.4f}")'''
        start_time = time.time()
        model.train()
        total_loss = 0
        for batch_idx, (text_inputs, text_lengths, audio_targets, audio_lengths) in enumerate(train_loader):
            '''if batch_idx == 0:
                for item in range(len(audio_targets)):
                    audio_target_np = audio_targets[item].detach().cpu().numpy()
                    #audio_pca_target_np = audio_pca_targets[item].detach().cpu().numpy()
                    plotFeatures(audio_target_np, '../../../../../ug/2021/etomlin/imgs/normal_'+str(item), save=True)
                    #plotFeatures(audio_pca_target_np, '../../../../../ug/2021/etomlin/imgs/pca_'+str(item), save=True)'''
            # move data to device
            text_inputs = text_inputs.to(device)
            text_lengths = text_lengths.to(device)
            audio_targets = audio_targets.to(device)
            audio_lengths = audio_lengths.to(device)
            optimiser.zero_grad()

            # run the model
            outputs = model(text_inputs, text_lengths, audio_targets, audio_lengths)

            # compute loss with masking
            max_len = audio_targets.size(1)
            mask = torch.arange(max_len, device=device)[None, :] < audio_lengths[:, None]
            mask = mask.unsqueeze(-1).expand_as(outputs)  # shape is [batch_size, max_len, num_mels]
            outputs_masked = outputs[mask]
            targets_masked = audio_targets[mask]
            loss = criterion(outputs_masked, targets_masked)

            # prep next loop
            loss.backward()
            # Apply gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimiser.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        time_passed = datetime.timedelta(seconds=(time.time() - start_time))
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {time_passed}')

        # step the lr with quick validation
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch_val_idx, (text_val_inputs, text_val_lengths, audio_val_targets, audio_val_lengths) in enumerate(val_loader):
                text_val_inputs = text_val_inputs.to(device)
                text_val_lengths = text_val_lengths.to(device)
                audio_val_targets = audio_val_targets.to(device)
                audio_val_lengths = audio_val_lengths.to(device)
                outputs = model(text_val_inputs, text_val_lengths, audio_val_targets, audio_val_lengths)
                loss = criterion(outputs, audio_val_targets)
                validation_loss += loss.item()

        avg_validation_loss = validation_loss / len(val_loader)
        print(f"Validation Loss: {avg_validation_loss:.4f}")
        scheduler.step(avg_validation_loss)
    #print(outputs)

def test(model, test_loader, criterion, optimiser):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():  # disable gradients
        for text_inputs, text_lengths, audio_targets, audio_lengths in test_loader:
            # move data to device
            text_inputs = text_inputs.to(device)
            text_lengths = text_lengths.to(device)
            audio_targets = audio_targets.to(device)
            audio_lengths = audio_lengths.to(device)

            # run the model
            outputs = model(text_inputs, text_lengths, audio_targets, audio_lengths)

            # compute loss with masking
            max_len = audio_targets.size(1)
            mask = torch.arange(max_len, device=device)[None, :] < audio_lengths[:, None]
            mask = mask.unsqueeze(-1).expand_as(outputs)  # shape is [batch_size, max_len, num_mels]
            outputs_masked = outputs[mask]
            targets_masked = audio_targets[mask]

            # compute loss
            loss = criterion(outputs_masked, targets_masked)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Average Test Loss: {avg_test_loss:.4f}')
    return avg_test_loss

def plotFeatures(data, data_name, save=False):  # note fft (dense sampling FT), hopsize, windowsize stuff for later
    plt.figure(figsize=(10,4))
    img = display.specshow(data, y_axis='log', x_axis='time', cmap='inferno')
    plt.title(data_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.yticks(np.array([numbers]), np.array([labels])) to change the values on the axis
    plt.colorbar(img, format="%+2.f dB")

    if (save):
        plt.savefig(data_name+'.png')
    else:
        plt.show()