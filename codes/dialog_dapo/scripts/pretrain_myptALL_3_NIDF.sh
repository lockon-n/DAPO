#! /bin/bash

# This script is for ubuntu-pretraining
# total samples: 1044590
# after executing
# --models/electraDAPO_myptALL_3_NIDF/--
# should have some models


### pre-work ###
export PROJECT_DIR=../../..
export model_name=electraDAPO_myptALL_3_NIDF
export TASK_NAME=rawtext_3_pretrain
export original_model_dir=google/electra-large-discriminator

### GPU parts start ###
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
### GPU parts ends  ###

### dialog_pretrain_process start ###
### totalnumber is 1044590 ###
export DATA_DIR=$PROJECT_DIR/datasets/dialog_eval_pretrain/rawtext_pretrain/
maxlen=512
gpubatchsize=10
savesteps=17410
logsteps=2000
evalsteps=17410
epoch=5
lr=1e-5
ws=8700

e1step=17410
e2step=34820
e3step=52230
e4step=69640
# if maxlen=128, then just a debug execution

export OUTPUT_DIR=$PROJECT_DIR/models/$model_name/
python ../pytorch_src/run_dialog_evaluation.py \
--model_name_or_path=$original_model_dir \
--task_name=$TASK_NAME --do_train --do_eval \
--data_dir=$DATA_DIR --max_seq_length=$maxlen \
--per_gpu_train_batch_size=$gpubatchsize \
--learning_rate=$lr --num_train_epochs=$epoch \
--output_dir=$OUTPUT_DIR --overwrite_output_dir \
--save_steps=$savesteps --eval_steps=$evalsteps \
--logging_steps=$logsteps --evaluate_during_training \
--warmup_steps=$ws

# process model output dir to get 5epochs models
cd $OUTPUT_DIR
mkdir epoch1
mkdir epoch2
mkdir epoch3
mkdir epoch4
mkdir epoch5

cp vocab.txt ./epoch1
cp vocab.txt ./epoch2
cp vocab.txt ./epoch3
cp vocab.txt ./epoch4

mv ./checkpoint-$e1step/* ./epoch1
mv ./checkpoint-$e2step/* ./epoch2
mv ./checkpoint-$e3step/* ./epoch3
mv ./checkpoint-$e4step/* ./epoch4
mv config.json special_tokens_map.json tokenizer_config.json training_args.bin pytorch_model.bin vocab.txt ./epoch5

rm -rf ./checkpoint-$e1step/
rm -rf ./checkpoint-$e2step/
rm -rf ./checkpoint-$e3step/
rm -rf ./checkpoint-$e4step/

# model dir ready

