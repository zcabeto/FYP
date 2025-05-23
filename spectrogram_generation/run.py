import argparse
import numpy as np
from rich.progress import track
import build
import train
from hparams import HParams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                        required=False, help='output path')
    parser.add_argument('-t', '--input_text', type=str, default=None,
                        required=False, help='path to input text')
    args = parser.parse_args()

    print("Loading Trained TTS Model...")
    hparams = HParams()
    model, criterion, optimiser = build.getModel(hparams)
    model, _, _ = train.load_checkpoint(args.checkpoint_path, model, None)

    with open(args.input_text, 'r') as f:
        lines = f.readlines()
        iterator = 0
        for line in track(lines[1:], description='Generating Spectrograms'):
            line_parts = line.split('\t')
            text, input_options = line_parts[0].strip(), line_parts[1:]
            mel_spectrogram, options = build.generate(text, model, input_options)
            np.save(f"{args.output_path}/spectrogram-{iterator}-{options['background']}-{options['quality']}.npy", mel_spectrogram)
            iterator += 1