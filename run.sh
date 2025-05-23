#!/bin/bash

# activate the running virtual environment
source spectrogram_generation/TTS_Venv/bin/activate

# run the python script for generating spectrograms
python spectrogram_generation/TextToSpec/run.py -o spectrogram_generation/spectrograms -c spectrogram_generation/model.pt -t inputs.tsv

# activate the venv_waveglow virtual environment
deactivate
source audio_generation/WaveGlow_Venv/bin/activate

# run the WaveGlow python script to convert spectrograms to audios
python audio_generation/run_waveglow.py -s spectrogram_generation/spectrograms -a audio_generation/audios

