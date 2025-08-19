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

        # --- Penambahan Lapisan CNN ---
        # CNN akan menerima input dengan dimensi (batch_size, sequence_length, features)
        # Kita perlu menganggap features sebagai 'channel' untuk CNN 1D
        # Atau jika input adalah gambar (2D), kita perlu CNN 2D.
        # Untuk kasus ini, asumsi input adalah data deret waktu dengan beberapa fitur (seperti kanal).
        # Misalnya, jika configs.enc_in adalah jumlah fitur, kita bisa anggap itu sebagai input channel.
        # Atau jika input adalah output embedding (d_model), kita bisa tambahkan CNN setelah embedding.

        # Kita akan menambahkan CNN setelah embedding untuk memproses fitur yang telah di-embed.
        # Dimensi input CNN: (batch_size, d_model, sequence_length) setelah transpose
        # Karena Conv1d mengharapkan (batch_size, channels, sequence_length)
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)
            # Anda bisa menambahkan lapisan Conv1d dan MaxPool1d lainnya di sini
            # nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=5, padding=2),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn_decoder = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # --- End Penambahan Lapisan CNN ---


        # LSTM perlu menyesuaikan input_size jika ada perubahan dimensi dari CNN
        # Output dari MaxPool1d akan mengurangi panjang urutan, jadi kita perlu perhatikan itu.
        # Untuk kesederhanaan, kita asumsikan output_channels dari CNN sama dengan d_model
        # dan kita akan menyesuaikan panjang urutan di forward pass.

        self.encoder_lstm = nn.LSTM(
            input_size=configs.d_model, # Input size tetap d_model jika CNN output d_model channels
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True
        )

        # Lapisan proyeksi perlu disesuaikan karena output Bi-LSTM adalah 2 * hidden_size
        self.projection = nn.Linear(configs.d_model * 2, configs.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Proses embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (batch_size, seq_len, d_model)
        dec_out = self.dec_embedding(x_dec, x_mark_dec) # (batch_size, label_len + pred_len, d_model)

        # Transpose untuk CNN: (batch_size, d_model, sequence_length)
        # CNN akan mengurangi panjang urutan.
        enc_out_cnn = self.cnn_encoder(enc_out.permute(0, 2, 1)).permute(0, 2, 1) # (batch_size, new_seq_len, d_model)
        dec_out_cnn = self.cnn_decoder(dec_out.permute(0, 2, 1)).permute(0, 2, 1) # (batch_size, new_dec_seq_len, d_model)

        # Proses Bi-LSTM
        # Inisialisasi hidden dan cell state dari encoder_lstm (biasanya nol atau dari mekanisme tertentu)
        # Jika tidak ada mekanisme spesifik, Bi-LSTM akan menginisialisasi sendiri
        _, (hidden, cell) = self.encoder_lstm(enc_out_cnn)

        # Decoder LSTM
        # Gunakan hidden dan cell dari encoder untuk memulai decoder
        # Pastikan dimensi hidden dan cell state sesuai dengan bidirectional: (num_layers * 2, batch_size, hidden_size)
        # Jika tidak sesuai, Anda mungkin perlu menyesuaikan atau menggunakan inisialisasi default
        dec_output, _ = self.decoder_lstm(dec_out_cnn, (hidden, cell)) # (batch_size, new_dec_seq_len, 2 * d_model)

        # Proyeksi ke output akhir
        output = self.projection(dec_output) # (batch_size, new_dec_seq_len, c_out)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} tidak didukung pada model ini.")