import argparse 
import os 
import json 
import logging

from transformers import AutoTokenizer
from tqdm import tqdm 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def main(args):

    hit = []
    if args.model_type == "OTTER":
        logger.info("Loading tokenizer for OTTER")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        logger.info("Loading dev dataset to measure hit@4k")
        with open(args.dev_reader_dataset_path, "r") as f:
            dev_reader_dataset = json.load(f)
        
        for datapoint in tqdm(dev_reader_dataset, desc="Measuring.."):
            qid = datapoint["question_id"]
            answer_text = datapoint["answer-text"]
            retrieved_tbs = datapoint["retrieved_tbs"]
            context_concated = ""
            for tb in retrieved_tbs:
                context_concated += tb["context"]
            
            answer_tokenized = tokenizer.tokenize(answer_text,truncation=True)
            context_tokenized = tokenizer.tokenize(context_concated, max_length=4096, truncation=True)

            answer_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokenized))
            context_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(context_tokenized))

            if answer_decoded.lower() in context_decoded.lower():
                hit.append(1)
            else:
                hit.append(0)
    else:
        # CORE or COS 
        logger.info(f"Loading tokenizer for {args.model_type}")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        logger.info("Loading dev dataset to measure hit@4k")
        with open(args.dev_reader_dataset_path, "r") as f:
            dev_reader_dataset = json.load(f)
        for datapoint in tqdm(dev_reader_dataset, desc="Measuring.."):
            # qid = datapoint["question_id"]
            answers = datapoint["answers"]
            assert len(answers) == 1
            answer_text = answers[0]

            retrieved_ctxs = datapoint["ctxs"]
            context_concated = ""
            for ctx in retrieved_ctxs:
                context_concated += (ctx["title"] + " [SEP] " + ctx["text"]) # Following the format of DPR
            
            answer_tokenized = tokenizer.tokenize(answer_text,truncation=True)
            context_tokenized = tokenizer.tokenize(context_concated, max_length=4096, truncation=True)

            answer_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokenized))
            context_decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(context_tokenized))

            if answer_decoded.lower() in context_decoded.lower():
                hit.append(1)
            else:
                hit.append(0)

    logger.info(f"Hit@4k: {sum(hit)/len(hit)}")

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["OTTER", "COS", "CORE"], default="OTTER")
    parser.add_argument("--dev_reader_dataset_path", type=str, default=None)

    args = parser.parse_args()
    main(args)