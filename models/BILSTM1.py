import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)  # Input is hidden_size, not 2 * hidden_size
        self.v = nn.Parameter(torch.rand(hidden_size))
    
    def forward(self, encoder_outputs):
        energy = torch.tanh(self.attn(encoder_outputs))
        attention_weights = F.softmax(torch.matmul(energy, self.v), dim=1)
        context = attention_weights.unsqueeze(-1) * encoder_outputs
        return context.sum(dim=1)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )

        num_lstm_layers = getattr(configs, 'e_layers', 2)
        lstm_dropout = getattr(configs, 'dropout', 0.1)

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )
        # Projection to reduce BiLSTM output to configs.d_model
        self.encoder_projection = nn.Linear(configs.d_model * 2, configs.d_model)
        self.encoder_norm = nn.LayerNorm(configs.d_model)

        # Attention
        self.attention = Attention(configs.d_model)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )
        self.decoder_norm = nn.LayerNorm(configs.d_model * 2)

        # Projection for output
        self.projection = nn.Linear(configs.d_model * 2, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder
        enc_input = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, (hidden, cell) = self.encoder_lstm(enc_input)
        enc_out = self.encoder_projection(enc_out)  # Project to configs.d_model
        enc_out = self.encoder_norm(enc_out + enc_input)  # Residual connection
        context = self.attention(enc_out)

        # Decoder
        dec_input = self.dec_embedding(x_dec, x_mark_dec)
        dec_out, _ = self.decoder_lstm(dec_input, (hidden, cell))
        dec_out = self.decoder_norm(dec_out)

        # Output projection
        output = self.projection(dec_out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} tidak didukung.")