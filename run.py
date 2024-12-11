import torch
import numpy as np
from pathlib import Path
from build import train, test, getData, getModel, plotFeatures, generate
from hparams import HParams

print('>>Initialising Data')
hparams = HParams(n=1, epochs=1)
root_dir = str(Path(__file__).resolve().parent.parent)
train_set, val_set, test_set = getData(hparams, use_existing_data=False)

print(f'>>Initialising Model')
model, criterion, optimiser = getModel(hparams)

for i in range(26):
    print(f">>Loading Checkpoint")
    #checkpoint = torch.load('model.pt')
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    
    print(f">>Training Model")
    model.train()
    train(model, train_set, val_set, criterion, optimiser, hparams)
    torch.save({'model_state_dict': model.state_dict(),'optimiser_state_dict': optimiser.state_dict()}, 'model.pt')

    print(f">>Generating Audio")
    model.eval()
    audio, spectrogram = generate("hello world", model)
    plotFeatures(spectrogram, f"{root_dir}/audios/spect", save=True)
    np.save(f"{root_dir}/audios/data.npy", audio)

print(f">>Testing Model")
model.eval()
test(model, test_set, criterion)
