"""
Train row classifier.
We will use this row classifier to filter out the rows that are not related to the question.
As a train and dev dataset, we use examples that have only one row containing the answer.

"""

import argparse 
import logging 
import os 
import random
import numpy as np
import wandb 
import torch
import pickle 
import evaluate 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)  
from transformers import set_seed
from datasets import Dataset, DatasetDict
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_datapoint_to_text(question, table, passages, meta_data):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = f'[TABLE] Title: {meta_data["title"]}\nSection Title: {meta_data["section_title"]}\n Data: ' + \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    
    text = f"{question} [SEP] {table_str} {passage_str}"
    return text 

def load_dataset(args):
    # Load dataset for training and evaluation
    with open(args.train_file, "rb") as f:
        train_file = pickle.load(f)
    with open(args.dev_file, "rb") as f:
        dev_file = pickle.load(f)
    
    train_data = []
    dev_data = []

    for datapoint in tqdm(train_file, desc="Converting train file to training dataset..."):
        pos_text = convert_datapoint_to_text(datapoint["question"], datapoint["table"], datapoint["passages"], datapoint["meta_data"])
        neg_text = convert_datapoint_to_text(datapoint["question"], datapoint["neg_table"], datapoint["neg_passages"], datapoint["meta_data"])

        train_data.append({"text": pos_text, "label": 1})
        train_data.append({"text": neg_text, "label": 0})

    for datapoint in tqdm(dev_file, desc="Converting dev file to dev dataset..."):
        pos_text = convert_datapoint_to_text(datapoint["question"], datapoint["table"], datapoint["passages"], datapoint["meta_data"])
        neg_text_bm25 = convert_datapoint_to_text(datapoint["question"], datapoint["bm25_neg_table"], datapoint["bm25_neg_passages"], datapoint["meta_data"])
        neg_text_random = convert_datapoint_to_text(datapoint["question"], datapoint["rd_neg_table"], datapoint["rd_neg_passages"], datapoint["meta_data"])
        
        dev_data.append({"text": pos_text, "label": 1})
        dev_data.append({"text": neg_text_bm25, "label": 0})
        dev_data.append({"text": neg_text_random, "label": 0})

    train_dataset  = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    dataset = DatasetDict({"train":train_dataset, "validation":dev_dataset})
    return dataset



def main(args):
    # Set seed
    wandb.login()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    effective_batch_size = args.effective_batch_size
    num_gpu_available = torch.cuda.device_count()
    print("Number of GPUs available: ", num_gpu_available)
    num_workers = int(4 * int(num_gpu_available))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info("Created output directory: {}".format(args.output_dir))

    grad_accumulation_steps = int(effective_batch_size / num_gpu_available /  args.per_device_train_batch_size)
    if grad_accumulation_steps * num_gpu_available * args.per_device_train_batch_size != effective_batch_size:
        raise ValueError("Given batch size configuration is not possible!")


    set_seed(args.seed)
    logger.info("Seed set to {}".format(args.seed))
    logger.info("Loading datasets for training and evaluation...")

    dataset = load_dataset(args)

    # Load tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accu_res = accuracy_metric.compute(predictions=predictions, references=labels)
        recall_res = recall_metric.compute(predictions=predictions, references=labels)
        f1_res = f1_metric.compute(predictions=predictions, references=labels)

        return {"accuracy": accu_res["accuracy"], "recall": recall_res["recall"], "f1": f1_res["f1"]}

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to="wandb",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        dataloader_num_workers=num_workers,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        logging_steps=args.logging_steps,
    )

    # Optiemizer: AdamW, learning rate scheduler: linear decay, weight_decay=0.01
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    if args.do_train_and_eval:
        trainer.train()
        best_model_save_path = os.path.join(args.output_dir, "best_model")
        trainer.save_model(best_model_save_path)

    if args.do_eval:
        trainer.evaluate(tokenized_dataset["validation"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                        default="../preprocessed_data/row_classifier/train_intable_bm25_blink_false_positive_removal.pkl")
    parser.add_argument("--dev_file", type=str,
                        default="../preprocessed_data/row_classifier/dev__blink_false_positive_removal.pkl")
    # Training arguments 
    parser.add_argument("--effective_batch_size", default=16, type=int)
    parser.add_argument("--per_device_train_batch_size", default=4, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=4, type=int)    
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    parser.add_argument("--max_seq_length", type=int, default=512)

    # Other arguments
    parser.add_argument("--do_train_and_eval", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    main(args)