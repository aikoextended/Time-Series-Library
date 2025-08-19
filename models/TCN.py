import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
import numpy as np


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation  # Causal padding
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        # Input shape: [B, C, L]
        out = self.conv1(x)  # [B, C', L]
        out = out[:, :, :-self.padding]  # Remove padding to maintain causality
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        residual = x if self.residual is None else self.residual(x)
        out = out + residual
        return out


class Model(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling
    Replaces Transformer with dilated causal convolutions
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        
        # TCN Layers
        self.tcn_layers = nn.ModuleList()
        num_layers = configs.e_layers  # Reuse e_layers for number of TCN blocks
        kernel_size = 3  # Standard kernel size for TCN
        for i in range(num_layers):
            dilation = 2 ** i  # Exponential dilation for increasing receptive field
            self.tcn_layers.append(
                TCNBlock(
                    in_channels=configs.d_model,
                    out_channels=configs.d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=configs.dropout
                )
            )
        
        # Output projection
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        
        # Layer normalization
        self.norm = nn.LayerNorm(configs.d_model)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        # x_enc: [B, L, D_enc_in], x_mark_enc: [B, L, D_mark]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, L, d_model]
        enc_out = enc_out.permute(0, 2, 1)  # [B, d_model, L] for Conv1d
        
        # TCN processing
        for tcn in self.tcn_layers:
            enc_out = tcn(enc_out)  # [B, d_model, L]
        
        enc_out = enc_out.permute(0, 2, 1)  # [B, L, d_model]
        enc_out = self.norm(enc_out)
        
        # Decoder embedding
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, L', d_model]
        
        # For forecasting, we use the encoder output directly
        # Assuming x_dec includes placeholder for prediction length
        out = self.projection(enc_out)  # [B, L, c_out]
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
        # Other tasks (imputation, anomaly_detection, classification) can be added
        # but require specific TCN adaptations
        return None