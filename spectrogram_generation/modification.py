import numpy as np
import torch

class AudioModification_AfterInference:
    '''Class to modify audio features based on the text input AFTER inference'''
    def pitch_change(spectrogram, steps):
        '''Change the pitch of an audio file by a given number of steps
        steps > 0 increases the pitch
        steps < 0 decreases the pitch
        '''
        shift = 2 ** (steps/12)
        freq_bins = spectrogram.shape[0]
        new_freqs = np.arange(freq_bins) * shift

        shifted_spectrogram = np.zeros_like(spectrogram)
        for t in range(spectrogram.shape[1]):
            shifted_spectrogram[:, t] = np.interp(
                np.arange(freq_bins),
                new_freqs,
                spectrogram[:, t],
                left=0,
                right=0
            )
        return shifted_spectrogram

    def speed_change(spectrogram, factor):
        '''Change the speed of an audio file by a given factor
        factor can be float or integer
        factor > 1 speeds up the audio
        factor < 1 slows down the audio
        '''
        if factor == 1:
            return spectrogram
            
        # Convert to float and calculate new length
        factor = float(factor)
        new_len = int(spectrogram.shape[1] / factor)
        
        # Use linear interpolation for smooth speed change
        old_indices = np.arange(spectrogram.shape[1])
        new_indices = np.linspace(0, spectrogram.shape[1]-1, new_len)
        
        return np.array([np.interp(new_indices, old_indices, spectrogram[i,:]) 
                        for i in range(spectrogram.shape[0])])

    def volume_change(audio, vol_factor):
        '''Change the volume of an audio file by a given factor
        vol_factor > 1 increases the volume
        vol_factor < 1 decreases the volume
        '''
        audio = audio * (1 / vol_factor)
        return audio

class AudioModification_DuringInference:
    '''Class to modify audio features based on the text input DURING inference'''
    def __init__(self, text_tensor, options, PADDING_LEN=6):
        self.text_tensor = text_tensor
        self.options = options
        first_jitter = np.random.uniform(0, self.options['jitter'])
        self.mods = {'intonation':(False, 0), 'stress': False, 'speed':(False, 1), 'volume':(False, 1), 'jitter': (False, first_jitter, 0)}
        self.modification_map = {
            'falling/intonation': -3,
            'rising/intonation': -4,
            'stress': -5,
            'up/speed': -6,
            'down/speed': -7,
            'up/volume': -8,
            'down/volume': -9,
            'pause': -10,
            'jitter': -11,
        }
        self.mel_idx = 0
        self.speed_changes = []
        self.PADDING_LEN = PADDING_LEN
        self.pause_length = options['pause_length']
    
    def mod_intonation(self, mel_output):
        mel_output_numpy = mel_output.float().cpu().detach().numpy().reshape(-1,1)
        modified = AudioModification_AfterInference.pitch_change(mel_output_numpy, self.mods['intonation'][1]).reshape(mel_output.shape[0], mel_output.shape[1])
        return torch.tensor(modified, device=mel_output.device)

    def mod_volume(self, mel_output):
        return AudioModification_AfterInference.volume_change(mel_output, self.mods['volume'][1])

    def mod_jitter(self, mel_output):
        if self.mods['jitter'][2] >= 2:     # reset jitter every 2 frames (as 1 frame too short)
            jit = np.random.uniform(0, self.options['jitter'])
            self.mods['jitter'] = (True, jit, 0)
        else:
            self.mods['jitter'] = (True, self.mods['jitter'][1], self.mods['jitter'][2] + 1)
        mel_output_numpy = mel_output.float().cpu().detach().numpy().reshape(-1,1)
        modified = AudioModification_AfterInference.pitch_change(mel_output_numpy, self.mods['jitter'][1]).reshape(mel_output.shape[0], mel_output.shape[1])
        return torch.tensor(modified, device=mel_output.device)
    
    def mod_stress(self, mel_output):
        prior_intonation = self.mods['intonation']
        prior_volume = self.mods['volume']
        self.mods['intonation'] = (True, 0.3)
        self.mods['volume'] = (True, 0.75)
        mel_output = self.mod_intonation(mel_output)
        mel_output = self.mod_volume(mel_output)
        self.mods['intonation'] = prior_intonation
        self.mods['volume'] = prior_volume
        return mel_output

    def mod_speed_at_end(self, mel_outputs):
        if len(self.speed_changes) == 0:
            return mel_outputs
        
        mel_output_numpy = mel_outputs.float().cpu().detach().numpy()
        final_result = []
        current_position = 0
        sorted_changes = sorted(self.speed_changes, key=lambda x: x[0])     # sort just in case
        
        for i in range(len(sorted_changes)):
            start_idx = sorted_changes[i][0]
            end_idx = sorted_changes[i][1] if sorted_changes[i][1] != -1 else mel_output_numpy.shape[2]
            speed_factor = sorted_changes[i][2]
            
            # previous unmodified segment added
            if start_idx > current_position:
                for batch in range(mel_output_numpy.shape[0]):
                    if len(final_result) == 0:
                        final_result = [mel_output_numpy[batch, :, current_position:start_idx]]
                    else:
                        final_result[batch] = np.concatenate([final_result[batch], mel_output_numpy[batch, :, current_position:start_idx]], axis=1)
            
            # partition segment to modify
            segment = mel_output_numpy[:, :, start_idx:end_idx]
            segment_length = end_idx - start_idx
            new_length = int(segment_length / speed_factor)
            
            # apply speed change to each mel channel & add
            for batch in range(mel_output_numpy.shape[0]):
                modified_segment = np.zeros((segment.shape[1], new_length))
                for mel_idx in range(segment.shape[1]):
                    modified_segment[mel_idx] = np.interp(
                        np.linspace(0, segment_length-1, new_length),
                        np.arange(segment_length),
                        segment[batch, mel_idx, :]
                    )
                if len(final_result) <= batch:
                    final_result.append(modified_segment)
                else:
                    final_result[batch] = np.concatenate([final_result[batch], modified_segment], axis=1)
            current_position = end_idx
        
        # remaining unmodified segment added
        if current_position < mel_output_numpy.shape[2]:
            for batch in range(mel_output_numpy.shape[0]):
                final_result[batch] = np.concatenate([final_result[batch], mel_output_numpy[batch, :, current_position:]], axis=1)
        
        return torch.tensor(np.stack(final_result), dtype=mel_outputs.dtype, device=mel_outputs.device)
    
    def process(self, mel_output):
        self.mel_idx += 1
        if self.mods['intonation'][0]:
            mel_output = self.mod_intonation(mel_output)
        if self.mods['volume'][0]:
            mel_output = self.mod_volume(mel_output)
        if self.mods['jitter'][0]:
            mel_output = self.mod_jitter(mel_output)
        if self.mods['stress']:
            mel_output = self.mod_stress(mel_output)

        if self.mods['speed'][0]:
            if self.speed_changes == []:
                self.speed_changes.append([self.mel_idx, -1, self.mods['speed'][1]])
            elif self.speed_changes[-1][1] != -1:
                self.speed_changes.append([self.mel_idx, -1, self.mods['speed'][1]])
        if not self.mods['speed'][0] and len(self.speed_changes)>0:
            if self.speed_changes[-1][1] == -1:
                self.speed_changes[-1][1] = self.mel_idx
        return mel_output

    def is_modification(self, modification, startEnd, char_idx):
        if modification == 'stress':
            if self.text_tensor[0][char_idx] == self.modification_map['stress'] \
                    or self.text_tensor[0][char_idx-1] == self.modification_map['stress']:
                return True
        elif startEnd == 'start':
            if self.text_tensor[0][char_idx] == self.modification_map[modification] \
                    and self.text_tensor[0][char_idx + 1] == -1:
                return True
            elif self.text_tensor[0][char_idx - 1] == self.modification_map[modification] \
                    and self.text_tensor[0][char_idx] == -1:
                return True
        elif startEnd == 'end':
            if self.text_tensor[0][char_idx] == self.modification_map[modification] \
                    and self.text_tensor[0][char_idx-1] == -2:
                return True
            elif self.text_tensor[0][char_idx+1] == self.modification_map[modification] \
                    and self.text_tensor[0][char_idx] == -2:
                return True
        return False

    def update_mods(self, char_idx):
        if self.mods['stress']:
            self.mods['stress'] = False     # end stressed syllable immediately if previously started
        if self.text_tensor[0][char_idx] >= 0 or char_idx == 0:
            pass    # regular characters no modification OR prevent -1 overflow on subsequent checks
        elif self.is_modification('falling/intonation', 'start', char_idx):
            self.mods['intonation'] = (True, -self.options['intonation_pitch'])    # falling intonation
        elif self.is_modification('rising/intonation', 'start', char_idx):
            self.mods['intonation'] = (True, self.options['intonation_pitch'])     # rising intonation
        elif self.is_modification('up/speed', 'start', char_idx):
            self.mods['speed'] = (True, self.options['intonation_speed'])         # speed up
        elif self.is_modification('down/speed', 'start', char_idx):
            self.mods['speed'] = (True, 1/self.options['intonation_speed'])         # slow down
        elif self.is_modification('up/volume', 'start', char_idx):
            self.mods['volume'] = (True, self.options['intonation_volume'])        # volume up
        elif self.is_modification('down/volume', 'start', char_idx):
            self.mods['volume'] = (True, 1/self.options['intonation_volume'])      # volume down
        elif self.is_modification('stress', None, char_idx):
            self.mods['stress'] = True                                  # single stressed syllable
        elif self.is_modification('jitter', 'start', char_idx):
            self.mods['jitter'] = (True, self.mods['jitter'][1], 0)     # add jitter
        elif self.is_modification('falling/intonation', 'end', char_idx) or self.is_modification('rising/intonation', 'end', char_idx):
            self.mods['intonation'] = (False, 0)     # regular intonation
        elif self.is_modification('up/speed', 'end', char_idx) or self.is_modification('down/speed', 'end', char_idx):
            self.mods['speed'] = (False, 1.0)         # regular speed
        elif self.is_modification('up/volume', 'end', char_idx) or self.is_modification('down/volume', 'end', char_idx):
            self.mods['volume'] = (False, 1.0)        # regular volume
        elif self.is_modification('jitter', 'end', char_idx):
            self.mods['jitter'] = (False, 0, 0)        # add jitter
        elif char_idx >= len(self.text_tensor[0])-self.PADDING_LEN:  # end of audio, close all modifications
            if self.mods['intonation'] == True:
                self.mods['intonation'] = (False, self.mods['intonation'][1])     # regular intonation
            if self.mods['speed'][0] == True:
                self.mods['speed'] = (False, 1.0)         # regular speed
            if self.mods['volume'][0] == True:
                self.mods['volume'] = (False, 1.0)        # regular volume
            if self.mods['jitter'][0] == True:
                self.mods['jitter'] = (False, 0)       # no jitter
                
    def no_mod_to_modification_idx(self):
        mapping = {}
        no_mod_idx, with_mod_idx = 0, 0
        for i in range(len(self.text_tensor[0])):
            if self.text_tensor[0][i] > 0:
                mapping[no_mod_idx] = with_mod_idx
                no_mod_idx += 1
            with_mod_idx += 1
        return mapping