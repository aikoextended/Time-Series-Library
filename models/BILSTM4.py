import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1 / (self.head_dim ** 0.5)
        self.q_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.k_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.out_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
    
    def forward(self, encoder_outputs):
        batch_size, seq_len, hidden = encoder_outputs.shape
        query = self.q_linear(encoder_outputs[:, -1, :]).view(batch_size, 1, self.num_heads, self.head_dim)
        keys = self.k_linear(encoder_outputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.v_linear(encoder_outputs).view(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        energy = torch.matmul(query, keys.transpose(-1, -2)) * self.scale
        attention_weights = F.softmax(energy, dim=-1)
        context = torch.matmul(attention_weights, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.num_heads * self.head_dim)
        return self.out_linear(context.squeeze(1))

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

        # Encoder CNN + LSTM
        self.enc_conv = nn.Conv1d(configs.enc_in, configs.d_model, kernel_size=5, padding=2)
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
        self.attention = MultiHeadAttention(configs.d_model)

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
        enc_input = self.enc_embedding(x_enc, x_mark_enc)
        enc_input = self.enc_embed_projection(enc_input)
        enc_out, (hidden, cell) = self.encoder_lstm(enc_input)
        enc_out = self.dropout(enc_out)
        enc_out = self.encoder_norm(enc_out + enc_input)
        context = self.attention(enc_out)

        # Decoder
        dec_input = self.dec_embedding(x_dec, x_mark_dec)
        dec_input = self.dec_embed_projection(dec_input)
        dec_out, _ = self.decoder_lstm(dec_input, (hidden, cell))
        dec_out = self.decoder_norm(dec_out + dec_input)  # Residual connection

        # Output projection
        output = self.projection(dec_out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} tidak didukung.")