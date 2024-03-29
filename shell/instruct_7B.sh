echo $1, $2
seed=$2
output_dir="./model"
base_model="../autodl-tmp/decapoda-research-llama-7B-hf"
train_data="./data/movie/train.json"
val_data="./data/movie/valid.json"
resume_from_checkpoint="./model/checkpoint-1000"
for lr in 1e-4; do
  for dropout in 0.05; do
    for sample in 64; do
      echo "lr: $lr, dropout: $dropout , seed: $seed, sample: $sample"
      CUDA_VISIBLE_DEVICES=$1 python -u finetune_rec.py \
        --base_model $base_model \
        --train_data_path $train_data \
        --val_data_path $val_data \
        --output_dir ${output_dir}_${seed}_${sample} \
        --batch_size 86 \
        --micro_batch_size 32 \
        --num_epochs 10 \
        --learning_rate $lr \
        --cutoff_len 512 \
        --lora_r 8 \
        --lora_alpha 16 --lora_dropout $dropout \
        --lora_target_modules '[q_proj,v_proj]' \
        --train_on_inputs \
        --group_by_length \
        --resume_from_checkpoint $resume_from_checkpoint \
        --sample $sample \
        --seed $2
    done
  done
done
