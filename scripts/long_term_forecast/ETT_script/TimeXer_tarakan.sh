export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path tarakan_weather_station_2023_cleaned.csv \
  --model_id TimeXer_tarakan \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1 \
  --target 'temp'