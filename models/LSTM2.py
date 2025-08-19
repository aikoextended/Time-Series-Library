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
        self.d_model = configs.d_model

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed,
            configs.freq, 
            configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, 
            configs.d_model, 
            configs.embed,
            configs.freq, 
            configs.dropout
        )

        num_lstm_layers = getattr(configs, 'e_layers', 1)
        lstm_dropout = getattr(configs, 'dropout', 0.1)

        # Encoder LSTM (unidirectional)
        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False
        )
        self.encoder_bn = nn.BatchNorm1d(configs.d_model)

        # Decoder LSTM (unidirectional)
        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=False
        )
        self.decoder_bn = nn.BatchNorm1d(configs.d_model)

        # Attention mechanism
        embed_dim = configs.d_model
        num_heads = configs.factor
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=configs.dropout
        )

        # Projection and residual connection
        self.projection = nn.Linear(configs.d_model, configs.c_out)
        self.residual_fc = nn.Linear(configs.d_model, configs.d_model)

        # Debugging: Log dimensions
        print(f"Initialized ImprovedLSTM: d_model={configs.d_model}, num_heads={num_heads}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # Shape: [batch, seq_len, d_model]
        print(f"enc_embedding output shape: {enc_out.shape}")
        enc_out, (hidden, cell) = self.encoder_lstm(enc_out)  # Shape: [batch, seq_len, d_model]
        print(f"encoder_lstm output shape: {enc_out.shape}")
        enc_out = self.encoder_bn(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        print(f"encoder_bn output shape: {enc_out.shape}")

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # Shape: [batch, dec_len, d_model]
        print(f"dec_embedding output shape: {dec_out.shape}")
        attn_out, _ = self.attention(dec_out, enc_out, enc_out)  # Shape: [batch, dec_len, d_model]
        print(f"attention output shape: {attn_out.shape}")
        dec_out = dec_out + attn_out  # Residual connection for attention
        print(f"dec_out after attention shape: {dec_out.shape}")

        dec_out, _ = self.decoder_lstm(dec_out, (hidden, cell))  # Shape: [batch, dec_len, d_model]
        print(f"decoder_lstm output shape: {dec_out.shape}")
        dec_out = self.decoder_bn(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        print(f"decoder_bn output shape: {dec_out.shape}")

        # Projection with residual connection
        residual = self.residual_fc(dec_out)  # Shape: [batch, dec_len, d_model]
        print(f"residual_fc output shape: {residual.shape}")
        output = self.projection(dec_out + residual)  # Shape: [batch, dec_len, c_out]
        print(f"projection output shape: {output.shape}")
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} not supported.")