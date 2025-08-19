import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        # Embedding untuk encoder dan decoder
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )

        num_lstm_layers = getattr(configs, 'e_layers', 2)
        lstm_dropout = getattr(configs, 'dropout', 0.0)

        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True  # Diubah menjadi Bi-directional
        )

        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            # Ukuran hidden_size tetap sama, namun outputnya akan 2 * hidden_size
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True  # Diubah menjadi Bi-directional
        )

        # Lapisan proyeksi perlu disesuaikan karena output Bi-LSTM adalah 2 * hidden_size
        # Output dari Bi-LSTM akan memiliki dimensi B x S x (2 * hidden_size)
        # Hidden state dan cell state akan memiliki dimensi (2 * num_layers) x B x hidden_size
        self.projection = nn.Linear(configs.d_model * 2, configs.c_out) # Dikalikan 2

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        _, (hidden, cell) = self.encoder_lstm(enc_out)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        dec_output, _ = self.decoder_lstm(dec_out, (hidden, cell))

        output = self.projection(dec_output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} tidak didukung pada model ini.")