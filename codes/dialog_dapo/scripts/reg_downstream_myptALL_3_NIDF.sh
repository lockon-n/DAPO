#! /bin/bash

# This script is for reg_myptALL_3_downstream

### pre-work ###
export PROJECT_DIR=../../..
export model_dir=electraDAPO_myptALL_3_NIDF
export ep=5

### GPU parts start ###
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
### GPU parts ends  ###

# parameters start
export gpubatchsize=10
export maxlen=512
export lr=1e-5
export epoch=8
export wd=0
export sd=42
# parameters end

## DD task 720 train examples
export OUTPUT_DIR=$PROJECT_DIR/results/$model_dir/dialog_electra_dd/
export DATA_DIR=$PROJECT_DIR/datasets/dialog_annotations/dd
export savesteps=12
export logsteps=5
export evalsteps=12
export ws=10
export TASK_NAME=dd_overall
python ../pytorch_src/run_dialog_evaluation.py \
--model_name_or_path=$PROJECT_DIR/models/$model_dir/epoch$ep/ \
--task_name=$TASK_NAME --do_train --do_eval --do_predict \
--data_dir=$DATA_DIR --max_seq_length=$maxlen \
--per_gpu_train_batch_size=$gpubatchsize --learning_rate=$lr \
--num_train_epochs=$epoch --output_dir=$OUTPUT_DIR --overwrite_output_dir \
--save_steps=$savesteps --eval_steps=$evalsteps --logging_steps=$logsteps \
--evaluate_during_training --warmup_steps=$ws --weight_decay=$wd --seed=$sd


## PC task 720 train examples
export OUTPUT_DIR=$PROJECT_DIR/results/$model_dir/dialog_electra_pc/
export DATA_DIR=$PROJECT_DIR/datasets/dialog_annotations/pc
export savesteps=12
export logsteps=5
export evalsteps=12
export ws=10
export TASK_NAME=pc_overall

python ../pytorch_src/run_dialog_evaluation.py \
--model_name_or_path=$PROJECT_DIR/models/$model_dir/epoch$ep/ \
--task_name=$TASK_NAME --do_train --do_eval --do_predict \
--data_dir=$DATA_DIR --max_seq_length=$maxlen \
--per_gpu_train_batch_size=$gpubatchsize --learning_rate=$lr \
--num_train_epochs=$epoch --output_dir=$OUTPUT_DIR --overwrite_output_dir \
--save_steps=$savesteps --eval_steps=$evalsteps --logging_steps=$logsteps \
--evaluate_during_training --warmup_steps=$ws --weight_decay=$wd --seed=$sd


## FED_fialog total 125 examples
export OUTPUT_DIR=$PROJECT_DIR/results/$model_dir/dialog_electra_fed_dialog/
export DATA_DIR=$PROJECT_DIR/datasets/dialog_annotations/fed_dialog
export savesteps=3
export logsteps=2
export evalsteps=3
export ws=3
for name in coherent errorrecovery consistent likeable understanding flexible informative inquisitive diverse depth overall
do
export TASK_NAME=fed_dialog_$name
python ../pytorch_src/run_dialog_evaluation.py \
--model_name_or_path=$PROJECT_DIR/models/$model_dir/epoch$ep/ \
--task_name=$TASK_NAME --do_eval --data_dir=$DATA_DIR --max_seq_length=$maxlen \
--per_gpu_train_batch_size=$gpubatchsize --learning_rate=$lr \
--num_train_epochs=$epoch --output_dir=$OUTPUT_DIR --overwrite_output_dir \
--save_steps=$savesteps --eval_steps=$evalsteps --logging_steps=$logsteps \
--evaluate_during_training
done
