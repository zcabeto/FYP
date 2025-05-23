class HParams:
    def __init__(self):
        ## RUNTIME PARAMETERS ##
        self.n = 13100
        self.training_n = 10480
        self.testing_n = 1965
        self.metadata_file = 'data_create/metadata.txt'
        self.epochs=10
        self.seed=1234
        self.learning_rate=1e-3
        self.batch_size=32

        ## AUDIO PARAMETERS ##
        self.max_wav=32768.0
        self.sr=22050
        self.filter_length=1024
        self.hop_length=256
        self.win_length=1024
        self.n_mels=80
        self.fmin=0.0
        self.fmax=8000.0

        ## MODEL PARAMETERS ##
        # Text Parameters
        self.n_symbols=148
        self.embedding_dim=512

        # Encoder parameters
        self.encoder_kernel=5
        self.encoder_dim=512

        # Decoder parameters
        self.decoder_dim=1024
        self.prenet_dim=256
        self.max_decoder_steps=1000
        self.threshold=0.5
        self.dropout=0.1

        # Attention parameters
        self.attention_rnn_dim=1024
        self.attention_dim=128
        self.attention_n_filters=32
        self.attention_kernel_size=31