## Final Year Project: 21041581
This repository relates to a UCL MEng Mathematical Computation FYP. It is a fully implemented TTS tool allowing user-inputted text to form playable audio, additionally providing capabilities to manually adjust various speech features of the whole generated audio or individual specific segments. A full description of usage is provided in the related paper, but input.tsv presents some example inputs.
This code may be freely copied and distributed, provided the source is explicitly acknowledged.

Found on GitHub: https://github.com/zcabeto/FYP

Make sure to get the rest of the data from Dropbox: https://www.dropbox.com/scl/fo/tnmka4n53mxf0lb5guzim/AIE9FhiVPS8m9dxpravmbyA?rlkey=au1s20o8eyj9qxym4syow3m8s&st=rxrtrmij&dl=0  
--> /audio_modification/background_noise  
--> /spectrogram_modification/model.pt

-------

# The Missing Link in Synthetic Voices: Exploring the Vocal Features Behind Cyberattack Deception
With recent improvements in AI, we confront the increasing use of deepfake voices in cyberattacks - but how much do we really know about them? With limited existing literature in the field, this paper aims to inspire a line of new research into how synthetic voices deceive victims in social engineering attacks. We launch this field with the creation of a synthetic speech tool configured for user-specified adjustment of various vocal characteristics, meant for use in investigating the properties of manipulative speech patterns. In creating this tool, we conducted a literature review and scoping review to establish the needs of this area and collect existing knowledge on how voices can be used to persuade and deceive. Subsequent ideas for research using this tool are discussed, plotting the future directions of this new field. By cultivating an understanding of synthetic cyberattack-intended speech, we aim to devise better and more knowledgeable preventative measures.

-------

### SETUP INSTRUCTIONS
From the root directory, two virtual environments require set-up. The project is made to run in Python 3.6.8, but newer Python3 versions should run the same. Your system should run with CUDA capabilities.

To form the environment for text-to-mel spectrogram conversion, the following virtual environment is needed. If a different environment name is used, change the run.sh script to reflect this. The requirements may optionally require updates to setuptools to install.
```console
cd spectrogram_generation
python3 -m venv TTS_Venv
source TTS_Venv/bin/activate
pip install -r requirements.txt
deactivate; cd ..
```

The spectrogram to audio conversion uses a pre-trained WaveGlow model using the following virtual environment. 
```console
cd audio_generation
python3 -m venv WaveGlow_Venv
source WaveGlow_Venv/bin/activate
pip install -r requirements.txt
deactivate; cd ..
```

The LJSpeech-1.1 dataset can be downloaded from https://keithito.com/LJ-Speech-Dataset/ to have the WAV files placed in /spectrogram_generation/LJSpeech-1.1/wavs. However, this is only needed for training, not for generation. For inference and generation, the trained model (model.pt) and the folder of background noise choices can be found on Dropbox via the link above.

### RUN INSTRUCTIONS
In input.tsv, each row should contain a text input, optionally including modification metacharacters, followed by tab-separated values for modification amounts. For full use, many inputs can be listed at once. Note that appending "_ filler" allows the full speech to be produced best, as the final syllable can be a little temperamental.

To run, having adjusted the run.sh file for system-specific processing and venv names, run
```console
./run
```

------

There is a chance that, upon running, an error in this imported model occurs just before audio generation. Namely:
```console
File "SYSTEM/.cache/torch/hub/NVIDIA_DeepLearningExamples_torchhub/PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py", line 28, in <module>
    torch._C._jit_set_autocast_mode(False)
AttributeError: module 'torch._C' has no attribute '_jit_set_autocast_mode'
```
If you encounter this issue, which is a result of WaveGlow being built before PyTorch 1.7, a simple solution is to change the cached /waveglow/model.py file to just delete this line.



