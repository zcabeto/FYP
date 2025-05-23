from statistics import mean, stdev
import os
import time
import argparse
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import torch
import time
import datetime
import build
from hparams import HParams
import data_create.audio_proc as AudioProcessor

def load_checkpoint(checkpoint_path, model, optimiser):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    if optimiser is not None:
        optimiser.load_state_dict(checkpoint_dict['optimiser'])
    return model, optimiser, 0


def save_checkpoint(model, optimiser, epoch_number, filepath):
    torch.save({'epoch': epoch_number,'state_dict': model.state_dict(),'optimiser': optimiser.state_dict()}, filepath)


def train(model, train_loader, val_loader, criterion, optimiser, hparams, output_directory, epoch_number):    
    torch.cuda.empty_cache()
    model.train()
    for epoch in range(epoch_number, hparams.epochs):
        print("Epoch {}".format(epoch))
        # validate before running epoch
        print(f"Validation Loss: {validate(model, val_loader, criterion):.4f}")
        start_time = time.time()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            # move data to device
            model.zero_grad()
            x = tuple(x_val.to(build.device) for x_val in batch[0])
            y =	tuple(y_val.to(build.device) for y_val in batch[1])

            # run the model
            y_pred = model(x)

            # compute loss with masking
            loss = criterion(y_pred, y)
            loss_item = max(loss.item(), 1e-8)
            print(f"Training Loss {i}/{len(train_loader)}: {loss_item:.4f}")

            # prep next loop + gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item()
            torch.cuda.empty_cache()

        time_passed = datetime.timedelta(seconds=(time.time() - start_time))
        print(f'Epoch [{epoch}/{hparams.epochs}], Loss: {total_loss:.4f}, Time: {time_passed}')

        checkpoint_path = os.path.join(output_directory, "checkpoint")
        save_checkpoint(model, optimiser, epoch+1, checkpoint_path)

def validate(model, val_loader, criterion):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            # move data to device
            x = tuple(x_val.to(build.device) for x_val in batch[0])
            y =	tuple(y_val.to(build.device) for y_val in batch[1])

            # run the model
            y_pred = model(x)

            # compute loss with masking
            loss = criterion(y_pred, y)
            loss_item = max(loss.item(), 1e-8)
            validation_loss += loss_item

            torch.cuda.empty_cache()
    avg_validation_loss = validation_loss / len(val_loader)
    model.train()
    return avg_validation_loss

def test(model, test_loader, criterion):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # move data to device
            x = tuple(x_val.to(build.device) for x_val in batch[0])
            y =	tuple(y_val.to(build.device) for y_val in batch[1])

            # run the model
            y_pred = model(x)

            # compute loss with masking
            loss = criterion(y_pred, y)
            loss_item = max(loss.item(), 1e-8)
            print(f"Test Loss {i}/{len(test_loader)}: {loss_item:.4f}")
            test_loss.append(loss_item)

            torch.cuda.empty_cache()
    print(f"Mean: {mean(test_loss):.4f}, Stdev: {stdev(test_loss):.4f}")
    model.train()
    return mean(test_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')

    args = parser.parse_args()
    hparams = HParams()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model, criterion, optimiser = build.getModel(hparams)
    if args.checkpoint_path is not None:
        model, optimiser, epoch_number = load_checkpoint(args.checkpoint_path, model, optimiser)
    else:
        epoch_number = 0
    
    train_loader, val_loader, test_loader = build.getData(hparams)
    AudioProcessor.setup_stft(hparams)
    
    print("Starting Training")
    train(model, train_loader, val_loader, criterion, optimiser, hparams, args.output_directory, epoch_number)
    test(model, test_loader, criterion)
