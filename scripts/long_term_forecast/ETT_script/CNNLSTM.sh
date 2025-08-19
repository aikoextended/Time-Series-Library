export CUDA_VISIBLE_DEVICES=0

model_name=CNNLSTM

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path tarakan_weather_station_2023_bersih.csv \
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
  --des 'bersih_temp_MS' \
  --itr 1 \
  --target 'temp' \
> "logs/CNNLSTM_bersih_temp_MS_96_$(date +'%Y%m%d_%H%M%S').log" 2>&1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path tarakan_weather_station_2023_bersih.csv \
  --model_id tarakan_96_192 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 256 \
  --batch_size 4 \
  --des 'bersih_temp_MS' \
  --itr 1 \
  --target 'temp' \
> "logs/CNNLSTM_temp_MS_192_$(date +'%Y%m%d_%H%M%S').log" 2>&1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path tarakan_weather_station_2023_bersih.csv \
  --model_id tarakan_96_336 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 256 \
  --batch_size 4 \
  --des 'bersih_temp_MS' \
  --itr 1 \
  --target 'temp' \
> "logs/CNNLSTM_temp_MS_336_$(date +'%Y%m%d_%H%M%S').log" 2>&1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/custom/ \
  --data_path tarakan_weather_station_2023_bersih.csv \
  --model_id tarakan_96_720 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 1 \
  --d_model 256 \
  --batch_size 4 \
  --des 'bersih_temp_MS' \
  --itr 1 \
  --target 'temp' \
> "logs/CNNLSTM_temp_MS_720_$(date +'%Y%m%d_%H%M%S').log" 2>&1