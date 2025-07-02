import numpy as np
import xgboost as xgb
import torch.nn as nn  # 🟢 penting

class XGBoostForecast:
    def __init__(self, configs):
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.models = []

    def fit(self, x_enc, y_true):
        B, seq_len, D = x_enc.shape
        _, pred_len, D_y = y_true.shape
        assert D == 1 and D_y == 1, "XGBoostForecast hanya mendukung data univariat"

        X = x_enc[:, :, 0]
        Y = y_true[:, :, 0]

        self.models = []
        for i in range(self.pred_len):
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X, Y[:, i])
            self.models.append(model)

    def forecast(self, x_enc):
        B, L, D = x_enc.shape
        assert D == 1, "Hanya input univariat"

        X = x_enc[:, :, 0]
        forecasts = []
        for model in self.models:
            pred = model.predict(X)
            forecasts.append(pred)
        forecasts = np.stack(forecasts, axis=1)
        return forecasts[:, :, np.newaxis]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            return self.forecast(x_enc)
        return None


# ✅ Tambahkan wrapper yang kompatibel dengan PyTorch
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = XGBoostForecast(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    # Override method .float() agar tidak error
    def float(self):
        return self  # tidak melakukan apapun, hanya untuk kompatibilitas
