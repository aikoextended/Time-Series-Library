import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding 
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        mlp_input_dim_after_embedding = self.seq_len * self.d_model

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.pred_len == 1:
                self.mlp = nn.Sequential(
                    nn.Linear(mlp_input_dim_after_embedding, 16),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(16, 16),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(16, self.c_out)
                )
            elif self.pred_len == 10:
                self.mlp = nn.Sequential(
                    nn.Linear(mlp_input_dim_after_embedding, 32),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(16, self.c_out * self.pred_len)
                )
            else: # Menggunakan arsitektur MLP umum untuk pred_len lainnya
                self.mlp = nn.Sequential(
                    nn.Linear(mlp_input_dim_after_embedding, self.d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model * 4, self.d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                )
                self.output_layer = nn.Linear(self.d_model, self.c_out * self.pred_len)
            
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)

        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim_after_embedding, configs.d_model * 4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 4, configs.d_model * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 2, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout)
            )
            self.output_layer = nn.Linear(configs.d_model, configs.c_out)
        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout_cls = nn.Dropout(configs.dropout)
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim_after_embedding, configs.d_model * 4),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 4, configs.d_model * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 2, configs.d_model),
                nn.ReLU(),
                nn.Dropout(configs.dropout)
            )
            self.output_layer = nn.Linear(configs.d_model, configs.num_class)
        else:
            raise ValueError(f"Nama tugas tidak didukung: {self.task_name}")

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        
        mlp_out = self.mlp(enc_out)

        # Menerapkan output_layer jika ada (untuk kasus pred_len umum)
        if hasattr(self, 'output_layer') and self.task_name in ['long_term_forecast', 'short_term_forecast'] and self.pred_len not in [1, 10]:
            mlp_out = self.output_layer(mlp_out)

        output = mlp_out.reshape(mlp_out.shape[0], self.pred_len, self.c_out)
        return output

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        mlp_out = self.mlp(enc_out)
        output = self.output_layer(mlp_out.unsqueeze(1).repeat(1, self.seq_len, 1))
        return output

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        mlp_out = self.mlp(enc_out)
        output = self.output_layer(mlp_out.unsqueeze(1).repeat(1, self.seq_len, 1))
        return output

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self.act(enc_out)
        enc_out = self.dropout_cls(enc_out) 
        if x_mark_enc is not None:
            enc_out = enc_out * x_mark_enc.unsqueeze(-1)
        
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
        mlp_out = self.mlp(enc_out)
        output = self.output_layer(mlp_out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
