import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers=1, cell_type='GRU'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        try:
            self.rnn = getattr(nn, cell_type)(embedding_dim, hidden_dim, num_layers, batch_first= True)
        except:
            raise ValueError("cell_type must be 'RNN', 'LSTM', or 'GRU'")
        self.cell_type = cell_type

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers=1, cell_type='GRU'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        try:
            self.rnn = getattr(nn, cell_type)(embedding_dim, hidden_dim, num_layers, batch_first=True)
        except:
            raise ValueError("cell_type must be 'RNN', 'LSTM', or 'GRU'")
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
        self.cell_type = cell_type

    def forward(self, tgt, hidden):
        embedded = self.embedding(tgt)
        outputs, hidden = self.rnn(embedded, hidden)
        logits = self.fc(outputs)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        encoder_outputs, hidden = self.encoder(src)
        # For LSTM, hidden is a tuple (h_n, c_n)
        decoder_hidden = hidden
        outputs, _ = self.decoder(tgt, decoder_hidden)
        return outputs
