BERT_BASE_DIR=./pretrain
rm -rf $BERT_BASE_DIR/pretraining_output
TF_CPP_MIN_LOG_LEVEL=1 CUDA_VISIBLE_DEVICES=0 \
python run_pretraining.py \
  --input_file=$BERT_BASE_DIR/tf_examples.tfrecord \
  --output_dir=$BERT_BASE_DIR/pretraining_output \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=$1 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=1000 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
