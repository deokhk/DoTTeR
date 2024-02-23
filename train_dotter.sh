export RUN_ID=0
export BASIC_PATH="."
export DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval
export TRAIN_DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval/train_intable_contra_blink_row_denoise.pkl
export DEV_DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval/dev_intable_contra_blink_row_denoise.pkl
export RT_MODEL_PATH=${BASIC_PATH}/model/trained_models/dotter
export PRETRAIN_MODEL_PATH=${BASIC_PATH}/trained_models/checkpoint-pretrain/checkpoint-87000/checkpoint_best.pt
export TABLE_CORPUS=table_corpus_blink
export SAVE_TENSOR_PATH=${BASIC_PATH}/preprocessed_data/retrieval/training/dotter
export RATE_MODEL_PATH=${BASIC_PATH}/model/trained_models/RATE/best_checkpoint
export ALL_BLINK_TABLE_PATH=${BASIC_PATH}/data_wikitable/all_constructed_blink_tables.json
export TOPK=15
export CONCAT_TBS=15
export SEED=42
export EXP_NAME=dotter
mkdir ${RT_MODEL_PATH}

# Here, effective batch size is train_batch_size
# For each gpu, per-device train batch size is train_batch_size / num_gpus / gradient_accumulation_steps

# We use A100-80GB X 4 GPUs to train the model
# For RTX-3090 (vram:24GB), per-device train batch size 4 is possible.

python -m scripts.train_1hop_tb_retrieval \
  --do_train \
  --prefix ${RUN_ID} \
  --predict_batch_size 100 \
  --model_name roberta-base \
  --all_blink_table_path ${ALL_BLINK_TABLE_PATH} \
  --shared_encoder \
  --train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --max_c_len 512 \
  --max_q_len 70 \
  --num_train_epochs 20 \
  --warmup_ratio 0.1 \
  --init_checkpoint ${PRETRAIN_MODEL_PATH} \
  --train_file ${TRAIN_DATA_PATH} \
  --predict_file ${DEV_DATA_PATH} \
  --output_dir ${RT_MODEL_PATH} \
  --inject_summary \
  --injection_scheme column \
  --rate_model_path ${RATE_MODEL_PATH}\
  --normalize_summary_table \
  --save_tensor_path ${SAVE_TENSOR_PATH} \
  --seed ${SEED}
