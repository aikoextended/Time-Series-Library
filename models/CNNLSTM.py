import torch
import torch.nn as nn
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

        # CNN Encoder sebelum LSTM
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM setelah CNN
        num_lstm_layers = getattr(configs, 'e_layers', 2)
        lstm_dropout = getattr(configs, 'dropout', 0.0)

        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0
        )

        # Proyeksi ke output (c_out = 1 jika MS target 'temperature')
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 1. Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, D]
        
        # 2. CNN butuh input [B, D, L]
        cnn_input = enc_out.permute(0, 2, 1)  # -> [B, D, L]
        cnn_out = self.cnn_encoder(cnn_input)  # [B, D, L]
        cnn_out = cnn_out.permute(0, 2, 1)     # -> [B, L, D]

        # 3. LSTM
        _, (hidden, cell) = self.lstm(cnn_out)

        # 4. Decoder embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        # 5. Decoder LSTM
        dec_output, _ = self.lstm(dec_out, (hidden, cell))

        # 6. Proyeksi
        output = self.projection(dec_output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} tidak didukung pada model ini.")
