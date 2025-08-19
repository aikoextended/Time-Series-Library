export CUDA_VISIBLE_DEVICES=0

model_name=BILSTM3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path tarakan_weather_station_2023_outliers_window_3.csv \
  --model_id tarakan_96_96 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 256 \
  --batch_size 4 \
  --des 'rolling_median_temp_MS_outliers_window_3' \
  --itr 1 \
  --target 'temp' \
> "logs/BiLSTM3_rolling_median_temp_MS_outliers_window_3_96_$(date +'%Y%m%d_%H%M%S').log" 2>&1