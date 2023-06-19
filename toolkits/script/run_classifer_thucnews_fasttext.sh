CURRENT_DIR=`pwd`
export EMBEDDING_DIR=$CURRENT_DIR/prev_trained_model/embedding
export DATA_DIR=$CURRENT_DIR/dataset
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="THUCNews"
#EMBEDDING_TYPE="embedding_SougouNews.npz"
EMBEDDING_TYPE="random"
python run.py \
  --model_type=FastText \
  --model_name_or_path=$EMBEDDING_DIR \
  --embedding_type=$EMBEDDING_TYPE \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --max_seq_length=32 \
  --per_gpu_train_batch_size=256 \
  --per_gpu_eval_batch_size=32 \
  --learning_rate=1e-3 \
  --num_train_epochs=20.0 \
  --logging_steps=100 \
  --save_steps=100 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
