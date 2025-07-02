import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

class Model:
    """
    Random Forest model adapted for time series forecasting, replacing iTransformer
    """

    def __init__(self, configs):
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_estimators = getattr(configs, 'n_estimators', 100)  # Number of trees
        self.max_depth = getattr(configs, 'max_depth', None)      # Max depth of trees
        self.random_state = getattr(configs, 'random_state', 42)  # For reproducibility
        
        # Initialize Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch_size, seq_len, num_features]
        # Convert torch tensors to numpy for Random Forest
        x_enc = x_enc.cpu().numpy() if isinstance(x_enc, torch.Tensor) else x_enc
        x_mark_enc = x_mark_enc.cpu().numpy() if isinstance(x_mark_enc, torch.Tensor) else x_mark_enc

        # Normalization (same as iTransformer)
        means = np.mean(x_enc, axis=1, keepdims=True)
        x_enc = x_enc - means
        stdev = np.sqrt(np.var(x_enc, axis=1, keepdims=True) + 1e-5)
        x_enc = x_enc / stdev

        batch_size, seq_len, num_features = x_enc.shape

        # Flatten input sequence for Random Forest
        # Combine x_enc and x_mark_enc if x_mark_enc is provided
        if x_mark_enc is not None:
            x_input = np.concatenate([x_enc, x_mark_enc], axis=-1)
        else:
            x_input = x_enc

        # Reshape to [batch_size, seq_len * num_features]
        x_input = x_input.reshape(batch_size, -1)

        # Fit model if not already trained
        if not hasattr(self.model, 'estimators_'):
            # Create dummy target for training (in practice, this should be actual target data)
            # For forecasting, we assume we're predicting future values based on input sequence
            y_dummy = np.zeros((batch_size, self.pred_len * num_features))
            self.model.fit(x_input, y_dummy)

        # Predict: Output shape [batch_size, pred_len * num_features]
        pred = self.model.predict(x_input)

        # Reshape predictions to [batch_size, pred_len, num_features]
        pred = pred.reshape(batch_size, self.pred_len, num_features)

        # De-Normalization
        pred = pred * (stdev[:, 0, :].reshape(batch_size, 1, num_features).repeat(self.pred_len, axis=1))
        pred = pred + (means[:, 0, :].reshape(batch_size, 1, num_features).repeat(self.pred_len, axis=1))

        return pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # Convert back to torch tensor for consistency
            dec_out = torch.from_numpy(dec_out).float() if isinstance(x_enc, torch.Tensor) else dec_out
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
        return None