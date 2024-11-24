import sounddevice as sd
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, lengths):
        x = self.embedding(x)                       # convert sequences to embeddings
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)  # loop sending the batch through the LSTM
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim, num_layers, num_mels):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=num_mels + 2 * encoder_hidden_dim,
            hidden_size=decoder_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention = nn.Linear(decoder_hidden_dim + 2 * encoder_hidden_dim, 1)
        self.fc = nn.Linear(decoder_hidden_dim, num_mels)
        self.relu = nn.ReLU()

    def forward(self, encoder_outputs, audio_targets, audio_lengths):
        batch_size, max_seq_len, _ = audio_targets.size()
        decoder_hidden = None  # init hidden state
        outputs = []
        # teacher forcing - use actual context of known info during training
        for t in range(max_seq_len):
            if decoder_hidden is not None:
                attn_weights = self.compute_attention(decoder_hidden[0][-1], encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            else: # first time step
                context = encoder_outputs.mean(dim=1, keepdim=True)
            
            decoder_input = torch.cat([audio_targets[:, t, :].unsqueeze(1), context], dim=2)
            # run the RNN
            output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)

            # generate prediction
            prediction = self.fc(output)
            outputs.append(prediction)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def compute_attention(self, hidden, encoder_outputs):
        # hidden is (batch_size, decoder_hidden_dim)
        # encoder_outputs is (batch_size, seq_len, 2*encoder_hidden_dim)
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(torch.cat((hidden, encoder_outputs), dim=2))
        attention = self.attention(energy).squeeze(2)
        attn_weights = torch.softmax(attention, dim=1)
        return attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        encoder.train()
        self.decoder = decoder
        decoder.train()

    def forward(self, text_inputs, text_lengths, audio_targets, audio_lengths):
        encoder_outputs, hidden, cell = self.encoder(text_inputs, text_lengths)
        outputs = self.decoder(encoder_outputs, audio_targets, audio_lengths)
        return outputs
    
    def generate(self, text_inputs):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            text_lengths = torch.tensor([text_inputs.size(1)], dtype=torch.long).to(text_inputs.device)
            encoder_outputs, hidden, cell = self.encoder(text_inputs, text_lengths)
            batch_size = text_inputs.size(0)
            num_mels = self.decoder.fc.out_features
            max_length = text_inputs.size(1) * 10
            outputs = []
            decoder_hidden = None  # init hidden state
            prev_output = torch.zeros(batch_size, 1, num_mels).to(text_inputs.device)
            for t in range(max_length):
                if decoder_hidden is not None:
                    attn_weights = self.decoder.compute_attention(decoder_hidden[0][-1], encoder_outputs)
                    context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                else: # first time step
                    context = encoder_outputs.mean(dim=1, keepdim=True)

                # run the RNN
                decoder_input = torch.cat([prev_output, context], dim=2)
                output, decoder_hidden = self.decoder.rnn(decoder_input, decoder_hidden)

                # generate prediction
                prediction = self.decoder.fc(output)
                outputs.append(prediction)

                prev_output = prediction
            outputs = torch.cat(outputs, dim=1)
        return outputs.squeeze(0).cpu().numpy().T
