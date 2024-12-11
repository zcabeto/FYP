class HParams:
    def __init__(self, n, epochs):
        # training hyperparameters
        self.n=n
        self.epochs=epochs

        # model hyperparameters
        self.batch_size = 32
        self.encoder_embedding_dim = 4900
        self.encoder_hidden_dim = 140
        self.decoder_hidden_dim = 140
        self.num_layers = 2
        self.learning_rate = 1e-3

        # data set distribution
        self.val_set_size = 0.05
        self.test_set_size = 0.2

        # text processing hyperparameters
        self.vocab_size = 70

        # audio processing hyperparameters
        self.n_mels = 256
        self.n_fft = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.sampling_rate = 22050
        self.mel_fmin=0.0
        self.mel_fmax=8000.0
        self.max_wav_value = 32768.0