# Denoising Table-Text Retrieval for Open-Domain Question Answering
This repository contains the source code for our LREC-COLING 2024 paper (to appear) "Denoising Table-Text Retrieval for Open-Domain Question Answering".

## Requirements

We provide script to create a conda environment with all the required packages. Make sure to have conda installed on your system.

Then, run the following command to create the environment.
```bash
conda create -n dotter python=3.10
conda activate dotter

sh create_env.sh
```

This codebase is built upon [OTTeR](https://github.com/Jun-jie-Huang/OTTeR). 
We follow the same data preprocessing steps as OTTeR, and provide the instruction mostly taken from OTTeR's README. For the rest of the README, we assume you are at the root of the repository, if not explicitly mentioned by "cd".

#### Step 0: Download dataset

##### Step0-1: OTT-QA dataset

```bash
git clone https://github.com/wenhuchen/OTT-QA.git
cp OTT-QA/release_data/* ./data_ottqa

mv OTT-QA/data/traindev_request_tok ./data_wikitable/
mv OTT_QA/data/traindev_tables_tok ./data_wikitable/
```

##### Step0-2: OTT-QA all tables and passages

```bash
cd data_wikitable/
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json
wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json
cd ../
```

##### Step0-3: Download fused block preprocessed from OTTeR
Download OTTeR's processed linked passage from [all_constructed_blink_tables.json](https://drive.google.com/drive/folders/1aQTOWdJ-khBm7x30y9w7LLTgT3tQ0xCy?usp=sharing). Then unzip it with `gunzip` and move the json file to `./data_wikitable`.

#### Step 1: Denoising OTT-QA dataset
To denoise the OTT-QA dataset, we need to train false-positive removal model.
Run the following command below to prepare the data for training the model.

##### Step1-1: Preprocess the data
```bash
mkdir ./preprocessed_data/
mkdir ./preprocessed_data/false_positive_removal
mkdir ./model/
mkdir ./model/trained_models

cd ./preprocessing
python false_positive_removal_preprocess.py --split train --nega intable_bm25 --aug_blink
python false_positive_removal_preprocess.py --split dev --aug_blink
```
Then it will make "train_intable_bm25_blink_false_positive_removal.pkl" and "dev__blink_false_positive_removal.pkl" in "./preprocessed_data/false_positive_removal". Let the path of former be `TRAIN_FILE` and the latter be `DEV_FILE`.

##### Step1-2: Train the model
Then, train the false positive removal model with the following command.

```bash
#!/bin/bash 
NUM_GPUS=2
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=${NUM_GPUS} ./scripts/train_row_classifier.py \
    --train_file ${TRAIN_FILE} \
    --dev_file ${DEV_FILE} \
    --seed 42 \
    --effective_batch_size 32 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --model_name_or_path bert-base-cased \
    --do_train_and_eval \
    --logging_steps 10 \
    --output_dir "./model/trained_models/false_positive_removal"
```
This will save the best model to `./model/trained_models/false_positive_removal/best_model`. Let the path of the best model be `MODEL_PATH`.

##### Step1-3: Denoise the OTT-QA dataset
```bash
mkdir ./preprocessed_data/retrieval
cd ./preprocessing
CUDA_VISIBLE_DEVICES=0 python retriever_preprocess.py --split train --nega intable_contra --aug_blink --denoise --denoise_model_path ${MODEL_PATH}
CUDA_VISIBLE_DEVICES=1 python retriever_preprocess.py --split dev --nega intable_contra --aug_blink --denoise --denoise_model_path ${MODEL_PATH}

```
This will make "train_intable_contra_blink_row_denoise.pkl" and "dev_intable_contra_blink_row_denoise.pkl" in "./preprocessed_data/retrieval". We denote the path of former as `DENOISED_TRAIN_FILE` and the latter as `DENOISED_DEV_FILE`.

#### Step 2: Training the rank-aware column encoder
```bash

python -m scripts.train_RATE \
    --num_train_steps 60000 \
    --evaluation_steps 1000 \
    --logging_steps 20 \
    --batch_size 32 \
    --evaluation_batch_size 128 \
    --wikitable_path ${WIKITABLE_PATH} \
    --output_dir ${OUTPUT_DIR} \
```
To train rank-aware column encoder, you need to specify the path to `./data_wikitable/all_plain_tables.json` as `WIKITABLE_PATH` and the output directory as `OUTPUT_DIR`.
We recommend `OUTPUT_DIR` to be an absolute path for `./model/trained_models/RATE`.


This will save the best model to `./model/trained_models/RATE/best_checkpoint`. Let the path of the best model be `RATE_MODEL_PATH`.

#### Step 3: Training the DoTTeR model (retriever)
##### Step3-1: Download synthetic-pretrained checkpoint from OTTeR
We initialize the encoder with the mixed-modality synthetic pretrained checkpoint from OTTeR.
Download the checkpoint from [here](https://drive.google.com/drive/folders/1aQTOWdJ-khBm7x30y9w7LLTgT3tQ0xCy). 
```
unzip -d ./checkpoint-pretrain checkpoint-pretrain.zip 
```
Then, move the ./checkpoint-pretrain to `./model/`.

##### Step3-2: Train the DoTTeR model
We provide a shell script to train the DoTTeR model.
Before running the script, you need to specify the path to the preprocessed data and the path to the RATE model.
```bash
sh train_dotter.sh
```
This will save the best model as `checkpoint_best.pt` in `RT_MODEL_PATH`.

##### Step 4: Evaluation

##### Step 4-1: Build retrieval corpus (fused blocks)
```bash
cd ./preprocessing
python corpus_preprocess.py
```
This will make "table_corpus_blink.pkl" in "./preprocessed_data/retrieval". 

##### Step 4-2: Encode corpus with the trained DoTTeR model

We first encode the OTT-QA dev set, then the table corpus(fused blocks) with the trained DoTTeR model.
```bash
export BASIC_PATH="."
export RATE_MODEL_PATH=${BASIC_PATH}/model/trained_models/RATE/best_checkpoint
export RT_MODEL_PATH=${BASIC_PATH}/model/trained_models/dotter

python -m scripts.encode_corpus \
  --do_predict \
  --predict_batch_size 100 \
  --model_name roberta-base \
  --shared_encoder \
  --predict_file ${BASIC_PATH}/data_ottqa/dev.json \
  --init_checkpoint ${RT_MODEL_PATH}/checkpoint_best.pt \
  --embed_save_path ${RT_MODEL_PATH}/indexed_embeddings/question_dev \
  --inject_summary \
  --injection_scheme "column" \
  --rate_model_path ${RATE_MODEL_PATH}\
  --normalize_summary_table \
  --max_c_len 512 \
  --num_workers 8

export DATA_PATH=${BASIC_PATH}/preprocessed_data/retrieval
export TABLE_CORPUS=table_corpus_blink

python -m scripts.encode_corpus \
    --do_predict \
    --encode_table \
    --shared_encoder \
    --predict_batch_size 800 \
    --model_name roberta-base \
    --predict_file ${DATA_PATH}/${TABLE_CORPUS}.pkl \
    --init_checkpoint ${RT_MODEL_PATH}/checkpoint_best.pt \
    --embed_save_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS} \
    --inject_summary \
    --injection_scheme "column" \
    --rate_model_path ${RATE_MODEL_PATH}\
    --normalize_summary_table \
    --max_c_len 512 \
    --num_workers 24
```

##### Step 4-3: Build index and search with FAISS

Table recall can be evaluated with the following command.
```bash
python -m scripts.eval_ottqa_retrieval \
	 --raw_data_path ${BASIC_PATH}/data_ottqa/dev.json \
	 --eval_only_ans \
	 --query_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/question_dev.npy \
	 --corpus_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
	 --id2doc_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
   --output_save_path ${RT_MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
   --beam_size 100
```
This will save the retrieval results to `dev_output_k100_${TABLE_CORPUS}.json` in `RT_MODEL_PATH/indexed_embeddings`.

Block recall can be evaluated with the following command, after evaluating table recall.

```bash
python -m scripts.eval_block_recall \
     --split dev \
     --retrieval_results_file ${RT_MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json
```

##### Step 4-4: Preparing QA dev data from retrieval outputs
This step will prepare the QA dev data from the retrieval outputs. We use the top 15 table-text blocks(fused blocks) for QA

```bash
export CONCAT_TBS=15
python -m preprocessing.qa_preprocess \
     --split dev \
     --topk_tbs ${CONCAT_TBS} \
     --retrieval_results_file ${RT_MODEL_PATH}/indexed_embeddings/dev_output_k100_${TABLE_CORPUS}.json \
     --qa_save_path ${RT_MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json
```

#### Step 5: QA model
##### Step 5-1: Prepare QA training data
This step will find the top 15 table-text blocks for each question in the training set using DoTTeR, and prepare the training data for the QA model.

```bash
export BASIC_PATH="."
export RATE_MODEL_PATH=${BASIC_PATH}/model/trained_models/RATE/best_checkpoint
export RT_MODEL_PATH=${BASIC_PATH}/model/trained_models/dotter
export TABLE_CORPUS=table_corpus_blink
export CONCAT_TBS=15

python -m scripts.encode_corpus \
  --do_predict \
  --predict_batch_size 100 \
  --model_name roberta-base \
  --shared_encoder \
  --predict_file ${BASIC_PATH}/data_ottqa/train.json \
  --init_checkpoint ${RT_MODEL_PATH}/checkpoint_best.pt \
  --embed_save_path ${RT_MODEL_PATH}/indexed_embeddings/question_train \
  --inject_summary \
  --injection_scheme "column" \
  --rate_model_path ${RATE_MODEL_PATH}\
  --normalize_summary_table \
  --max_c_len 512 \
  --num_workers 16

python -m scripts.eval_ottqa_retrieval \
	   --raw_data_path ${BASIC_PATH}/data_ottqa/train.json \
	   --eval_only_ans \
	   --query_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/question_train.npy \
	   --corpus_embeddings_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}.npy \
	   --id2doc_path ${RT_MODEL_PATH}/indexed_embeddings/${TABLE_CORPUS}/id2doc.json \
	   --output_save_path ${RT_MODEL_PATH}/indexed_embeddings/train_output_k100_${TABLE_CORPUS}.json \
	   --beam_size 100

python ../preprocessing/qa_preprocess.py \
	    --split train \
	    --topk_tbs ${CONCAT_TBS} \
	    --retrieval_results_file ${RT_MODEL_PATH}/indexed_embeddings/train_output_k100_${TABLE_CORPUS}.json \
	    --qa_save_path ${RT_MODEL_PATH}/train_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json
```


##### Step 5-2: Train the QA model
We use the same training script from OTTeR to train the QA model.

```bash
export BASIC_PATH="."
export TABLE_CORPUS=table_corpus_blink
export MODEL_NAME=mrm8488/longformer-base-4096-finetuned-squadv2
export RT_MODEL_PATH=${BASIC_PATH}/model/trained_models/dotter
export QA_MODEL_PATH=${BASIC_PATH}/trained_models/qa_longformer_${TOPK}/dotter
export CONCAT_TBS=15
export SEED=42
export TOPK=15
export EXP_NAME=dotter_qa

mkdir ${QA_MODEL_PATH}
python -m scripts.train_final_qa \
    --do_train \
    --do_eval \
    --model_type longformer \
    --dont_save_cache \
    --overwrite_cache \
    --model_name_or_path ${MODEL_NAME} \
    --evaluate_during_training \
    --data_dir ${RT_MODEL_PATH} \
    --output_dir ${QA_MODEL_PATH} \
    --train_file ${RT_MODEL_PATH}/train_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
    --dev_file ${RT_MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_seq_length 4096 \
    --doc_stride 1024 \
    --topk_tbs ${TOPK} \
    --seed ${SEED} \
    --run_name ${EXP_NAME} \
    --eval_steps 2000
```
In this script, we don't support setting effective batch size. Instead, we set the batch size per GPU and the number of GPUs. We use batch size 16 and 4 GPUs in the example above.

##### Step 5-3: Evaluting the QA model

```bash
export PREDICT_OUT=dotter_qa_dev_result
export MODEL_NAME=mrm8488/longformer-base-4096-finetuned-squadv2
export TOPK=15
export QA_MODEL_PATH=${BASIC_PATH}/trained_models/qa_longformer_${TOPK}/dotter

python -m scripts.train_final_qa \
    --do_predict \
    --model_type ${MODEL_NAME} \
    --dont_save_cache \
    --overwrite_cache \
    --model_name_or_path ${MODEL_NAME} \
    --data_dir ${RT_MODEL_PATH} \
    --output_dir ${QA_MODEL_PATH} \
    --predict_file ${RT_MODEL_PATH}/dev_preprocessed_${TABLE_CORPUS}_k100cat${CONCAT_TBS}.json \
    --predict_output_file ${PREDICT_OUT}.json \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 8 \
    --doc_stride 1024 \
    --topk_tbs ${TOPK} \
    --threads 4 \

```


## Acknowledgments
This codebase is built upon the codebase from [OTTeR](https://github.com/Jun-jie-Huang/OTTeR).
We thank authors for open-sourcing them.
