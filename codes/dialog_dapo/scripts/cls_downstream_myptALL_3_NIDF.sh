#! /bin/bash

# This script is for cls_myptALL_3_downstream

### pre-work ###
export PROJECT_DIR=../../..
export model_dir=electraDAPO_myptALL_3_NIDF
export ep=5

### GPU parts start ###
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
### GPU parts ends  ###

# parameters start
export maxlen=512
export lr=1e-5
export epoch=8
export wd=0
export sd=42

# parameters end

## MUTUAL task 7088 train examples
export OUTPUT_DIR=$PROJECT_DIR/results/$model_dir/dialog_electra_mutual/
export DATA_DIR=$PROJECT_DIR/datasets/dialog_response_selection/mutual/
export savesteps=394
export logsteps=50
export evalsteps=394
export ws=316
export gpubatchsize=3

export TASK_NAME=mutual

python ../pytorch_src/run_multiple_choice.py \
--model_name_or_path=$PROJECT_DIR/models/$model_dir/epoch$ep/ \
--task_name=$TASK_NAME --do_train --do_eval --do_predict \
--data_dir=$DATA_DIR --max_seq_length=$maxlen \
--per_gpu_train_batch_size=$gpubatchsize --learning_rate=$lr \
--num_train_epochs=$epoch --output_dir=$OUTPUT_DIR \
--overwrite_output_dir --save_steps=$savesteps --eval_steps=$evalsteps \
--logging_steps=$logsteps --evaluate_during_training \
--warmup_steps=$ws --weight_decay=$wd --seed=$sd


## MUTUAL_plus task 7088 train examples
export OUTPUT_DIR=$PROJECT_DIR/results/$model_dir/dialog_electra_mutual_plus/
export DATA_DIR=$PROJECT_DIR/datasets/dialog_response_selection/mutual_plus/
export savesteps=394
export logsteps=50
export evalsteps=394
export ws=316
export gpubatchsize=3

export TASK_NAME=mutual_plus

python ../pytorch_src/run_multiple_choice.py \
--model_name_or_path=$PROJECT_DIR/models/$model_dir/epoch$ep/ \
--task_name=$TASK_NAME --do_train --do_eval --do_predict \
--data_dir=$DATA_DIR --max_seq_length=$maxlen \
--per_gpu_train_batch_size=$gpubatchsize --learning_rate=$lr \
--num_train_epochs=$epoch --output_dir=$OUTPUT_DIR \
--overwrite_output_dir --save_steps=$savesteps --eval_steps=$evalsteps \
--logging_steps=$logsteps --evaluate_during_training \
--warmup_steps=$ws --weight_decay=$wd --seed=$sd