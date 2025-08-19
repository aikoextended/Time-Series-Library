import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from torch.nn.utils import weight_norm

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.scale = 1 / (hidden_size ** 0.5)
    
    def forward(self, encoder_outputs):
        query = encoder_outputs[:, -1, :]  # [batch_size, 2 * hidden_size]
        keys = encoder_outputs  # [batch_size, seq_len, 2 * hidden_size]
        energy = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)) * self.scale
        attention_weights = F.softmax(energy, dim=-1)
        context = torch.bmm(attention_weights, encoder_outputs)
        return context.squeeze(1)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                        padding=padding, dilation=dilation))
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        x = self.conv(x)[:, :, :-self.conv.padding[0]]  # Causal padding
        x = self.relu(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1, 2, 4, 8]):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList([
            TCNBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size, d)
            for i, d in enumerate(dilations)
        ])
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        for layer in self.layers:
            x = layer(x)
        return x + residual  # Residual connection

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
        self.enc_embed_projection = nn.Linear(configs.d_model, configs.d_model * 2)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        self.dec_embed_projection = nn.Linear(configs.d_model, configs.d_model * 2)

        num_lstm_layers = getattr(configs, 'e_layers', 1)
        lstm_dropout = getattr(configs, 'dropout', 0.2)

        # Encoder TCN
        self.enc_tcn = TCN(configs.enc_in, configs.d_model, kernel_size=3, dilations=[1, 2, 4, 8])
        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model * 2,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.encoder_norm = nn.LayerNorm(configs.d_model * 2)
        self.dropout = nn.Dropout(lstm_dropout)

        # Attention
        self.attention = Attention(configs.d_model)

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model * 2,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        self.decoder_norm = nn.LayerNorm(configs.d_model * 2)

        # Projection for output
        self.projection = nn.Linear(configs.d_model * 2, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder
        enc_input = self.enc_embedding(x_enc, x_mark_enc)  # [batch_size, seq_len, d_model]
        enc_input = self.enc_tcn(enc_input.transpose(1, 2)).transpose(1, 2)  # [batch_size, seq_len, d_model]
        enc_input = self.enc_embed_projection(enc_input)  # [batch_size, seq_len, 2 * d_model]
        enc_out, (hidden, cell) = self.encoder_lstm(enc_input)
        enc_out = self.dropout(enc_out)
        enc_out = self.encoder_norm(enc_out + enc_input)
        context = self.attention(enc_out)

        # Decoder
        dec_input = self.dec_embedding(x_dec, x_mark_dec)
        dec_input = self.dec_embed_projection(dec_input)
        dec_out, _ = self.decoder_lstm(dec_input, (hidden, cell))
        dec_out = self.decoder_norm(dec_out)

        # Output projection
        output = self.projection(dec_out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [batch_size, pred_len, c_out]
        else:
            raise NotImplementedError(f"Task {self.task_name} tidak didukung.")

