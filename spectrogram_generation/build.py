import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_create.text_proc import textFeatures
import data_create.audio_proc as AudioProcessor
import data_create.dataset as ds
from model import Seq2Seq
from modification import AudioModification_AfterInference as AudioModification
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')


def getData(hparams):
    train_dataset, val_dataset, test_dataset = ds.create_dataset(hparams)

    train_loader = DataLoader(train_dataset, num_workers=1, shuffle=True,
                              sampler=None,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, sampler=None, num_workers=1,
                                shuffle=False, batch_size=hparams.batch_size,
                                pin_memory=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, sampler=None, num_workers=1,
                                shuffle=False, batch_size=hparams.batch_size,
                                pin_memory=False, collate_fn=collate_fn)
    AudioProcessor.setup_stft(hparams)
    return train_loader, val_loader, test_loader

def getModel(hparams):
    model = Seq2Seq(hparams).to(device)
    optimiser = optim.Adam(model.parameters(), lr=hparams.learning_rate)
    return model, criterion, optimiser

def criterion(model_output, mel_target):
    mel_target = mel_target[0] # tuple to tensor
    mel_target.requires_grad = False

    # match dimensions as needed
    model_output = model_output.transpose(1, 2)
    if mel_target.dim() == 2:
        mel_target = mel_target.unsqueeze(0)
    mel_target = mel_target.transpose(1, 2)
    if model_output.size() != mel_target.size():
        min_batch = min(model_output.size(0), mel_target.size(0))
        min_time = min(model_output.size(1), mel_target.size(1))
        min_mels = min(model_output.size(2), mel_target.size(2))
        
        model_output = model_output[:min_batch, :min_time, :min_mels]
        mel_target = mel_target[:min_batch, :min_time, :min_mels]
    
    # mask for padded regions
    mask = (torch.sum(torch.abs(mel_target), dim=1) > 1e-5).float().unsqueeze(1)
    masked_output = model_output * mask
    masked_target = mel_target * mask

    # compute loss
    criterion = nn.MSELoss().to(device)
    return criterion(masked_output, masked_target)

def collate_fn(batch):
    # make padded text
    input_lengths, ids = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
    text_padded = torch.LongTensor(len(batch), input_lengths[0])
    text_padded.zero_()
    for i in range(len(ids)):
        text = batch[ids[i]][0]
        text_padded[i, :text.size(0)] = text

    # make padded spectrogram
    num_mels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])
    mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
    mel_padded.zero_()
    for i in range(len(ids)):
        mel = batch[ids[i]][1]
        mel_padded[i, :, :mel.size(1)] = mel

    # format input x and output y
    x = (text_padded, input_lengths.data, mel_padded)
    y = (mel_padded)
    return x, y

def parse_options(options):
    parsed = {
        'overall_pitch': 0, 'overall_speed': 1, 'overall_volume': 1, 
        'intonation_pitch': 0, 'intonation_speed': 1, 'intonation_volume': 1, 
        'jitter': 0, 'pause_length': 5,
        'quality': 1, 'background': 'city_ambient-0.2'}
    if options == []: return parsed # return default values
    try:
        parsed['overall_pitch'] = float(options[0])
        parsed['overall_speed'] = float(options[1])
        parsed['overall_volume'] = float(options[2])
        parsed['intonation_pitch'] = float(options[3])
        parsed['intonation_speed'] = float(options[4])
        parsed['intonation_volume'] = float(options[5])
        parsed['jitter'] = float(options[6])
        parsed['pause_length'] = int(options[7])
        parsed['quality'] = float(options[8])
        parsed['background'] = options[9].strip()
    finally:
        return parsed               # if any misisng, use default values

def generate(text, model, options):
    options = parse_options(options)
    sequence = np.array(textFeatures(text))[None, :]
    sequence = torch.from_numpy(sequence).cuda().long()

    mel_outputs = model.inference(sequence, options)
    mel_spectrogram = mel_outputs.float().data.cpu().numpy()[0]
    mel_spectrogram = AudioModification.pitch_change(mel_spectrogram, options['overall_pitch'])
    mel_spectrogram = AudioModification.speed_change(mel_spectrogram, options['overall_speed'])
    mel_spectrogram = AudioModification.volume_change(mel_spectrogram, options['overall_volume'])
    return mel_spectrogram, options
