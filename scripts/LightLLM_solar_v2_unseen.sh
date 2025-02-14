model_name=LightLLM
train_epochs=3
learning_rate=0.01
llama_layers=32

master_port=50001
num_process=1
batch_size=1
d_model=32
d_ff=128

comment='LightLLM-Solar-V2-Unseen'


accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/solar/ \
  --data_path solar20Apr1_v2.csv \
  --val_path solar20Apr2_v2.csv \
  --model_id Solar_v2_Unseen \
  --model $model_name \
  --data solar_v2_unseen \
  --features MS \
  --seq_len 21 \
  --label_len 21  \
  --pred_len 21 \
  --factor 3 \
  --enc_in 18 \
  --dec_in 18 \
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
