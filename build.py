import sounddevice as sd
import torch
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch):
    text_inputs, audio_targets = zip(*batch)
    
    # convert lists to tensors
    text_inputs = [torch.tensor(t, dtype=torch.long) for t in text_inputs]
    text_lengths = torch.tensor([len(t) for t in text_inputs], dtype=torch.long)
    
    audio_targets = [torch.tensor(a.T, dtype=torch.float) for a in audio_targets]
    audio_lengths = torch.tensor([len(a) for a in audio_targets], dtype=torch.long)
    
    # pad sequences
    text_inputs_padded = pad_sequence(text_inputs, batch_first=True, padding_value=0)
    audio_targets_padded = pad_sequence(audio_targets, batch_first=True, padding_value=0.0)
    
    return text_inputs_padded, text_lengths, audio_targets_padded, audio_lengths


def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (text_inputs, text_lengths, audio_targets, audio_lengths) in enumerate(train_loader):
            # move data to device
            text_inputs = text_inputs.to(device)
            text_lengths = text_lengths.to(device)
            audio_targets = audio_targets.to(device)
            audio_lengths = audio_lengths.to(device)
            optimizer.zero_grad()

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
            optimizer.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def test(model, device, test_loader, criterion, optimizer):
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