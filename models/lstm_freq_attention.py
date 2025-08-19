import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class freq_att(nn.Module):
    """
    Frequency-domain multi-head attention using Fourier Transform
    """
    def __init__(self, dim_input, num_head, dropout):
        super(freq_att, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
        self.laynorm = nn.LayerNorm([dim_input])

    def forward(self, x):
        # x: [batch, seq_len, dim_input]
        # Apply FFT on the seq_len dimension (dim=1)
        q = torch.fft.fft(self.dropout(self.query(x)), dim=1)  # [batch, seq_len, dim_input]
        k = torch.fft.fft(self.dropout(self.key(x)), dim=1)    # [batch, seq_len, dim_input]
        v = torch.fft.fft(self.dropout(self.value(x)), dim=1)  # [batch, seq_len, dim_input]
        k = k.transpose(-2, -1)  # [batch, dim_input, seq_len]

        q_real = q.real
        k_real = k.real
        v_real = v.real
        q_imag = q.imag
        k_imag = k.imag
        v_imag = v.imag

        result_real = 0.0
        result_imag = 0.0
        for i in range(self.num_head):
            line_real = self.softmax(q_real @ k_real) @ v_real  # [batch, seq_len, seq_len] @ [batch, seq_len, dim_input]
            line_real = line_real.unsqueeze(-1)  # [batch, seq_len, dim_input, 1]
            line_imag = self.softmax(q_imag @ k_imag) @ v_imag  # [batch, seq_len, seq_len] @ [batch, seq_len, dim_input]
            line_imag = line_imag.unsqueeze(-1)  # [batch, seq_len, dim_input, 1]
            if i == 0:
                result_real = line_real
                result_imag = line_imag
            else:
                result_real = torch.cat([result_real, line_real], dim=-1)
                result_imag = torch.cat([result_imag, line_imag], dim=-1)

        # Combine real and imaginary parts via magnitude
        result = torch.sqrt(result_real * result_real + result_imag * result_imag)  # [batch, seq_len, dim_input, num_head]
        result = self.dropout(self.linear1(result).squeeze(-1)) + x  # [batch, seq_len, dim_input]
        result = self.laynorm(result)
        return result

class Model(nn.Module):
    """
    Pure 4-layer LSTM model with decomposition and Frequency Attention for time series forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_head = getattr(configs, 'num_head', 4)  # Default to 4 heads if not specified
        self.kernel_size = getattr(configs, 'kernel_size', 25)  # Kernel size for moving average decomposition

        # Simple linear embedding to map input features to d_model
        self.embedding = nn.Linear(configs.enc_in, configs.d_model)

        # 4-layer LSTM for trend and seasonal components
        self.lstm_trend = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=4,
            batch_first=True,
            dropout=configs.dropout if configs.dropout > 0 else 0,
            bias=True
        )
        self.lstm_seasonal = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=4,
            batch_first=True,
            dropout=configs.dropout if configs.dropout > 0 else 0,
            bias=True
        )

        # Frequency attention layers for trend and seasonal components
        self.attention_trend = freq_att(configs.d_model, self.num_head, configs.dropout)
        self.attention_seasonal = freq_att(configs.d_model, self.num_head, configs.dropout)

        # Projection layer to map combined attention output to prediction length
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def _decompose(self, x):
        # Moving average for trend component
        kernel_size = self.kernel_size
        padding = kernel_size // 2
        trend = F.avg_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=1, padding=padding)
        trend = trend.transpose(1, 2)  # [batch, seq_len, enc_in]
        
        # Seasonal component: input - trend
        seasonal = x - trend  # [batch, seq_len, enc_in]
        
        return trend, seasonal

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, N = x_enc.shape

        # Decomposition
        trend, seasonal = self._decompose(x_enc)  # [batch, seq_len, enc_in]

        # Embedding for trend and seasonal components
        trend_enc = self.embedding(trend)  # [batch, seq_len, d_model]
        seasonal_enc = self.embedding(seasonal)  # [batch, seq_len, d_model]

        # LSTM processing for trend and seasonal components
        trend_out, _ = self.lstm_trend(trend_enc)  # [batch, seq_len, d_model]
        seasonal_out, _ = self.lstm_seasonal(seasonal_enc)  # [batch, seq_len, d_model]

        # Frequency attention for trend and seasonal components
        trend_context = self.attention_trend(trend_out)  # [batch, seq_len, d_model]
        seasonal_context = self.attention_seasonal(seasonal_out)  # [batch, seq_len, d_model]

        # Combine trend and seasonal contexts by averaging
        context = (trend_context + seasonal_context) / 2  # [batch, seq_len, d_model]
        context = context.mean(dim=1)  # [batch, d_model]

        # Projection to prediction length
        dec_out = self.projection(context).unsqueeze(1)  # [batch, 1, pred_len]
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]  # [batch, pred_len, N]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        return None