from math import sqrt
import torch
import torch.nn.functional as F
from modification import AudioModification_DuringInference as AudioModification

SPACE_PADDING = [11, 11, 11, 11, 11, 11, 11]
## AUXILLIARY CLASSES ##
class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        # xavier initialization with a gain based on the chosen non-linearity
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation, bias=bias)
        # xavier initialization with a gain based on the chosen non-linearity
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, audio_signal):
        return self.conv(audio_signal)


## ENCODE THE EMBEDDED INPUTS ##
class Encoder(torch.nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        # build a list of convolutional layers with batch normalisation
        convolutions = []
        for _ in range(3):
            conv_layer = torch.nn.Sequential(
                ConvNorm(hparams.encoder_dim,
                         hparams.encoder_dim,
                         kernel_size=hparams.encoder_kernel,
                         stride=1,
                         padding=int((hparams.encoder_kernel - 1) / 2),
                         dilation=1,
                         w_init_gain='relu'),
                torch.nn.BatchNorm1d(hparams.encoder_dim)
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        # bidirectional LSTM (hidden size is half the embedding dimension per direction)
        self.lstm = torch.nn.LSTM(hparams.encoder_dim, int(hparams.encoder_dim / 2), 1, batch_first=True, bidirectional=True)

    def forward(self, embedded_input, input_lengths):
        # apply each convolution, followed by ReLU and dropout
        for conv in self.convolutions:
            embedded_input = F.dropout(F.relu(conv(embedded_input)), 0.5, self.training)
        # transpose to shape (batch, time, features) for LSTM processing
        embedded_input = embedded_input.transpose(1, 2)

        # pack the padded sequence for LSTM
        input_lengths = input_lengths.cpu().numpy()
        embedded_input = torch.nn.utils.rnn.pack_padded_sequence(embedded_input, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(embedded_input)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, embedded_input):
        # same as forward but without packing
        for conv in self.convolutions:
            embedded_input = F.dropout(F.relu(conv(embedded_input)), 0.5, self.training)
        embedded_input = embedded_input.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(embedded_input)
        return outputs

## DECODE TO SPECTROGRAM OUTPUTS ##
class Decoder(torch.nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mels = hparams.n_mels
        self.prenet_dim = hparams.prenet_dim
        self.encoder_dim = hparams.encoder_dim
        self.decoder_dim = hparams.decoder_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.threshold = hparams.threshold
        self.dropout = hparams.dropout

        # prenet setup as a sequence of LinearNorm layers
        self.layers = torch.nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for in_size, out_size in zip([hparams.n_mels] + [hparams.prenet_dim],[hparams.prenet_dim, hparams.prenet_dim])
        ])

        # alignment LSTM cell processes prenet output with encoder context
        self.attention_rnn = torch.nn.LSTMCell(hparams.prenet_dim + hparams.encoder_dim, hparams.attention_rnn_dim)

        # decoder LSTM cell processes context from the attention
        self.decoder_rnn = torch.nn.LSTMCell(hparams.attention_rnn_dim + hparams.encoder_dim, hparams.decoder_dim, 1)

        # linear projection generates a mel output
        self.linear_projection = LinearNorm(hparams.decoder_dim + hparams.encoder_dim, hparams.n_mels)

        # gate predicts when to stop decoding / end of audio
        self.gate_layer = LinearNorm(hparams.decoder_dim + hparams.encoder_dim, 1, bias=True, w_init_gain='sigmoid')

        # attention components
        self.query_layer = LinearNorm(hparams.attention_rnn_dim, hparams.attention_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(hparams.encoder_dim, hparams.attention_dim, bias=False, w_init_gain='tanh')
        self.v = LinearNorm(hparams.attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")

        # location-based attention convolution
        self.location_conv = ConvNorm(2, hparams.attention_n_filters,
                                      kernel_size=hparams.attention_kernel_size,
                                      padding=int((hparams.attention_kernel_size - 1) / 2),
                                      bias=False, stride=1, dilation=1)
        self.location_dense = LinearNorm(hparams.attention_n_filters, hparams.attention_dim, bias=False, w_init_gain='tanh')

    def attention_layer(self, attention_hidden_state, memory, processed_memory,
                        attention_weights_cat, mask):
        # get scores & features from previous weights
        processed_query = self.query_layer(attention_hidden_state.unsqueeze(1))
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        attention_sum = processed_query + processed_attention + processed_memory
        attention_sum = torch.clamp(attention_sum, -20, 20)     # prevent extreme values for NaN
        attention_scores = self.v(torch.tanh(attention_sum)).squeeze(-1)

        if mask is not None:
            attention_scores.data.masked_fill_(mask, self.score_mask_value)

        # normalise attention_scores for weights & context
        attention_weights = F.softmax(attention_scores + 1e-8, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)

        return attention_context, attention_weights

    def decode(self, decoder_input, memory, mask):
        # input + attention context --> update attentions
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, memory, self.processed_memory, attention_weights_cat, mask)

        # accumulate weights & update decoder LSTM
        self.attention_weights_cum += self.attention_weights
        decoder_input_combined = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input_combined, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.dropout, self.training)

        # make final predictions
        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction

    def forward(self, memory, decoder_inputs, memory_lengths):
        # initialise inputs + reshaping to prepare
        decoder_input = memory.data.new(memory.size(0), self.n_mels).zero_().unsqueeze(0)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), decoder_inputs.size(1), -1)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)

        # prenet processing
        for linear in self.layers:
            decoder_inputs = F.dropout(F.relu(linear(decoder_inputs)), p=0.5, training=True)

        # initialise & prepare states for decoding
        max_len = torch.max(memory_lengths).item()
        mask = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (mask < memory_lengths.unsqueeze(1)).bool()
        self.prepare_states(memory)

        mel_outputs = []
        # decode at each time step
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, _ = self.decode(decoder_input, memory, mask)
            mel_outputs.append(mel_output.squeeze(1))

        # reshape for output usage consistency
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mels).transpose(1, 2)
        return mel_outputs

    def inference(self, memory, original_text_tensor, options):
        self.prepare_states(memory)     # initialise & prepare states for decoding

        # prepare modifier to apply during inference
        modifier = AudioModification(original_text_tensor, options)
        idx_mapping = modifier.no_mod_to_modification_idx()

        # decode at each time step, leaving open for gate's prediction
        decoder_input = memory.data.new(memory.size(0), self.n_mels).zero_()
        mel_outputs = []
        char_idx, char_idx_no_mod = 0, 0
        while True:
            # run through prenet
            for linear in self.layers:
                decoder_input = F.dropout(F.relu(linear(decoder_input)), p=0.5, training=True)
            mel_output, gate_output = self.decode(decoder_input, memory, mask=None)

            # apply audio modifications
            if len(mel_outputs) > 0:
                _, focus_char = torch.max(self.attention_weights, dim=1)
                char_idx_no_mod_new = focus_char[0].item()
                if char_idx_no_mod_new != char_idx_no_mod:
                    modifier.update_mods(idx_mapping[char_idx_no_mod_new]-1)    # update mods for new char
                    # update mods for all chars in between if focus_char skips any
                    id_range = range(idx_mapping[char_idx_no_mod], idx_mapping[char_idx_no_mod_new]) if char_idx_no_mod_new > char_idx_no_mod else range(idx_mapping[char_idx_no_mod_new], idx_mapping[char_idx_no_mod])
                    for char_idx in id_range:
                        # iterate through all chars so none missed
                        modifier.update_mods(char_idx-1)
                        if original_text_tensor[0][char_idx-1] == modifier.modification_map['pause']:
                            mel_outputs += [mel_output] * modifier.pause_length
                    char_idx_no_mod = char_idx_no_mod_new
            mel_output = modifier.process(mel_output)           # apply pitch, volume, jitter, or emphasis during

            mel_outputs.append(mel_output.squeeze(1))

            if torch.sigmoid(gate_output.data) > self.threshold or char_idx >= len(original_text_tensor[0])-len(SPACE_PADDING)+1:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("reached the maximum steps - terminating early")
                break
            decoder_input = mel_output
        # reshape for output usage consistency
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mels).transpose(1, 2)
        mel_outputs = modifier.mod_speed_at_end(mel_outputs)    # apply speed mod at end
        return mel_outputs
    
    def prepare_states(self, memory):
        self.attention_hidden = memory.data.new(memory.size(0), self.attention_rnn_dim).zero_()
        self.attention_cell = memory.data.new(memory.size(0), self.attention_rnn_dim).zero_()
        self.decoder_hidden = memory.data.new(memory.size(0), self.decoder_dim).zero_()
        self.decoder_cell = memory.data.new(memory.size(0), self.decoder_dim).zero_()
        self.attention_weights = memory.data.new(memory.size(0), memory.size(1)).zero_()
        self.attention_weights_cum = memory.data.new(memory.size(0), memory.size(1)).zero_()
        self.attention_context = memory.data.new(memory.size(0), self.encoder_dim).zero_()
        self.processed_memory = self.memory_layer(memory)
        

## COMBINE ENCODER AND DECODER ##
class Seq2Seq(torch.nn.Module):
    def __init__(self, hparams):
        super(Seq2Seq, self).__init__()
        self.n_mels = hparams.n_mels

        # embedding, encoding, decoding layers
        self.embedding = torch.nn.Embedding(hparams.n_symbols, hparams.embedding_dim)
        val = sqrt(3.0) * sqrt(2.0 / (hparams.n_symbols + hparams.embedding_dim))
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

        # postnet convolutional layers to refine output
        in_channels = hparams.n_mels
        out_channels = hparams.encoder_dim
        pad = int((hparams.encoder_kernel - 1) / 2)
        kern = hparams.encoder_kernel

        self.convolutions = torch.nn.ModuleList()
        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(in_channels, out_channels, kernel_size=kern, stride=1,
                         padding=pad, dilation=1, w_init_gain='tanh'),
                torch.nn.BatchNorm1d(out_channels)
            )
        )
        for _ in range(1, 4):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(out_channels, out_channels, kernel_size=kern, stride=1,
                             padding=pad, dilation=1, w_init_gain='tanh'),
                    torch.nn.BatchNorm1d(out_channels)
                )
            )
        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(out_channels, in_channels, kernel_size=kern, stride=1,
                         padding=pad, dilation=1, w_init_gain='linear'),
                torch.nn.BatchNorm1d(in_channels)
            )
        )

    def forward(self, inputs):
        text_inputs, text_lengths, mel_targets = inputs

        # embed text inputs & encode
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        # decode to make mel_spectrograms & postnet refinement
        mel_outputs = self.decoder(encoder_outputs, mel_targets, memory_lengths=text_lengths)
        
        if torch.isnan(mel_outputs).any():
            average = torch.mean(mel_outputs)
            mel_outputs = torch.nan_to_num(mel_outputs, nan=average)
                
        refined_mel_outputs = mel_outputs.clone()
        for i in range(len(self.convolutions) - 1):
            refined_mel_outputs = F.dropout(torch.tanh(self.convolutions[i](refined_mel_outputs)), 0.5, self.training)
        refined_mel_outputs = F.dropout(self.convolutions[-1](refined_mel_outputs), 0.5, self.training)
        return mel_outputs + refined_mel_outputs

    def inference(self, inputs, options):
        inputs = add_spaces(inputs)     # add space padding
        embedded_inputs = self.embedding(remove_mods(inputs)).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs = self.decoder.inference(encoder_outputs, inputs, options)
        refined_mel_outputs = mel_outputs.clone()
        for i in range(len(self.convolutions) - 1):
            refined_mel_outputs = F.dropout(torch.tanh(self.convolutions[i](refined_mel_outputs)), 0.5, self.training)
        refined_mel_outputs = F.dropout(self.convolutions[-1](refined_mel_outputs), 0.5, self.training)
        return mel_outputs + refined_mel_outputs

def remove_mods(inputs):
    inputs_list = []
    for i in range(len(inputs[0])):
        if inputs[0][i] >= 0:
            inputs_list.append(inputs[0][i].item())
    inputs_no_mod = torch.tensor([inputs_list], dtype=inputs.dtype, device=inputs.device)
    return inputs_no_mod

def add_spaces(inputs):
    inputs_list = []
    for i in range(len(inputs[0])):
        inputs_list.append(inputs[0][i].item())
    inputs_list += SPACE_PADDING
    return torch.tensor([inputs_list], dtype=inputs.dtype, device=inputs.device)