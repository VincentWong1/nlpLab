CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/albert_base_zh
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="THUCNews"
python run.py \
  --model_type=albert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=32 \
  --per_gpu_train_batch_size=256 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=1e-5 \
  --num_train_epochs=5.0 \
  --logging_steps=1000 \
  --save_steps=1000 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
