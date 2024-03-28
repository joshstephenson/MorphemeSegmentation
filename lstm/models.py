import torch
import torch.nn as nn
import random
import os
from config import Config
from helpers import get_logger

config = Config()
logger = get_logger()

def init_weights(m):
    initial = config['preprocessing']['initial_weights']
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -initial, initial)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden_dim = config['encoder_decoder']['hidden_dim']
        self.n_layers = config['encoder_decoder']['n_layers']
        self.dropout = nn.Dropout(config['encoder_decoder']['dropout'])
        self.embedding = nn.Embedding(input_dim, config['encoder_decoder']['embedding_dim'])
        self.rnn = nn.LSTM(config['encoder_decoder']['embedding_dim'], config['encoder_decoder']['hidden_dim'],
                           config['encoder_decoder']['n_layers'],
                           bidirectional=config['encoder_decoder']['bidirectional'])

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights

class Decoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = config['encoder_decoder']['hidden_dim']
        self.n_layers = config['encoder_decoder']['n_layers']
        self.dropout = nn.Dropout(config['encoder_decoder']['dropout'])
        self.embedding = nn.Embedding(output_dim, config['encoder_decoder']['embedding_dim'])
        bidirectional = config['encoder_decoder']['bidirectional']
        self.rnn = nn.LSTM(config['encoder_decoder']['embedding_dim'], config['encoder_decoder']['hidden_dim'],
                           config['encoder_decoder']['n_layers'],
                           bidirectional=bidirectional)
        self.attention = SelfAttentionLayer(256)
        self.fc_out = nn.Linear(config['encoder_decoder']['hidden_dim'] * (2 if bidirectional else 1), output_dim)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(output_dim)
        self.device = device
        assert (
                self.encoder.hidden_dim == self.decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
                self.encoder.n_layers == self.decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

    def save(self):
        torch.save(self.state_dict(), config.model_file)
        logger.info(f'saved model to: {config.model_file}')

    def load_from_file(self, file):
        if not os.path.exists(file):
            logger.exception(file)
            logger.exception("No model file found. Perhaps you forgot to train first?")
            exit(1)
        try:
            self.load_state_dict(torch.load(file))
            logger.info(f"Loaded model from: {file}")
        except Exception as _:
            # we load this on the cpu when the model was generated on the hosted env
            self.load_state_dict(torch.load(file, map_location=torch.device('cpu')))