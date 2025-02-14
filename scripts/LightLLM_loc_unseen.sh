model_name=LightLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=50001
num_process=1
batch_size=1
d_model=32
d_ff=128

comment='LightLLM-Loc-Unseen'


accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/localization/ \
  --data_path data-night2-i_v2.csv \
  --val_path data-night5-i_v2.csv \
  --model_id Loc_Unseen \
  --model $model_name \
  --data loc_unseen \
  --features MS \
  --seq_len 28 \
  --label_len 28  \
  --pred_len 28 \
  --factor 3 \
  --enc_in 27 \
  --dec_in 27 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --target 'value'
