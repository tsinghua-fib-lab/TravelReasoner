##!/bin/bash

###############################chain-of-trips#########################################
# #####################################DeepSeek-R1-Distill-Llama-8B---B-SFT######################################
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path ./model/basic/deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset train_travel_reasoning_data_v11_large_select_2000 \
  --dataset_dir ./data \
  --template deepseekr1 \
  --finetuning_type lora \
  --lora_target q_proj,v_proj,k_proj,o_proj,up_proj,down_proj,gate_proj \
  --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
  --output_dir ./saves/DeepSeek-R1-Distill-Llama-8B/lora/large_2000/sft_all_v11 \
  --overwrite_output_dir \
  --cutoff_len 4096 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --warmup_ratio 0.03 \
  --save_strategy steps \
  --save_steps 200 \
  --eval_steps 100 \
  --do_eval \
  --eval_strategy steps \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --val_size 0.1 \
  --plot_loss \
  --save_total_limit 3 \
  --bf16


# #####################################DeepSeek-R1-Distill-Llama-8B----E-SFT######################################
CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train \
  --stage sft \
  --do_train \
  --model_name_or_path ./model/lora/v11_large_2000/DeepSeek-R1-Distill-Llama-8B-trained \
  --dataset train_travel_reasoning_data_v11_1000_200_enhance \
  --dataset_dir ./data \
  --template deepseekr1 \
  --finetuning_type lora \
  --lora_target q_proj,v_proj,k_proj,o_proj,up_proj,down_proj,gate_proj \
  --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
  --output_dir ./saves/DeepSeek-R1-Distill-Llama-8B/lora/large_2000/sft_all_v11_enhance \
  --overwrite_output_dir \
  --cutoff_len 4096 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --warmup_ratio 0.03 \
  --save_strategy steps \
  --save_steps 200 \
  --eval_steps 100 \
  --do_eval \
  --eval_strategy steps \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --val_size 0.1 \
  --plot_loss \
  --save_total_limit 3 \
  --bf16
  
