# Evaluate BM25 retrieval results

import os 
import argparse 
import logging
import json 
import pickle
import pandas 
from rank_bm25 import BM25Okapi
from typing import List 
from tqdm import tqdm
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def linearize_block(table, passages, meta_data):
    # normed text by processing table with " H1 is C1 .... "
    # We remove special tokens
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = meta_data['title']+ meta_data['section_title']+ \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    passage_str =  ' '.join(passages)
    return table_str + passage_str




def fill_index(data):

    for doc in tqdm(data):
        psg = doc['passages'][:8] # top 8 passages, following the encode_datasets.py for other models
        linearized_fused_block = linearize_block(doc['table'], psg, doc['meta_data'])
        yield {
            "_index" : "fused_blocks",
            "block" : linearized_fused_block,
            "table_id" : doc["table_id"]
        }


def evaluate_retriever(retriever, dev_data, block_to_tableid):
    
    # We retrieve once (top-100) and check if the table is hit.

    top_100_blocks = []
    dev_data = dev_data[0:10]
    for datapoint in tqdm(dev_data):
        question = datapoint["question"]
        top_n_blocks = retriever.query_index(question, 100)
        top_100_blocks.append(top_n_blocks)
    
    for top_k in [1,10,15,20,50,100]:
        table_hit = []
        block_hit = []
        for datapoint, top_100_block in tqdm(zip(dev_data, top_100_blocks)):
            question = datapoint["question"]
            gold_table_id = datapoint["table_id"]
            answer_text = datapoint["answer-text"]

            top_k_retrieved_block = top_100_block[:top_k]
            is_table_hit = False 
            is_block_hit = False
            for retrieved_block in top_k_retrieved_block:
                table_id = block_to_tableid[retrieved_block]["table_id"]
                if table_id == gold_table_id:
                    is_table_hit = True 
                    if answer_text.lower() in retrieved_block.lower():
                        is_block_hit = True
                        break
            if is_table_hit:
                table_hit.append(1)
            else:
                table_hit.append(0)
            
            if is_block_hit:
                block_hit.append(1)
            else:
                block_hit.append(0)
        
        logger.info(f"BM25 Retrieval Top-{top_k} Table Hit: {(sum(table_hit)/len(table_hit)) * 100}")
        logger.info(f"BM25 Retrieval Top-{top_k} Block Hit: {(sum(block_hit)/len(block_hit)) * 100}")
    
        
def main(args):
    logger.info(f"Loading linked tables from {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    es = Elasticsearch(timeout=30, max_retries=10, retry_on_timeout=True)
    bulk(es, fill_index(data))


    with open(args.dev_path, 'r') as f:
        dev_data = json.load(f)
        
    for top_k in [1,10,15,20,50,100]:
        table_hit = []
        block_hit = []

        for datapoint in tqdm(dev_data):
            question = datapoint["question"]
            gold_table_id = datapoint["table_id"]
            answer_text = datapoint["answer-text"]

            is_table_hit = False 
            is_block_hit = False

            query = {
                "query": {
                    "match": {
                        "block" : question
                    }
                }
            }
            res = es.search(index="fused_blocks", body=query, size=top_k)
            matched_blocks = res["hits"]["hits"]
    
            for block in matched_blocks:
                block_content = block["_source"]["block"]
                block_table_id = block["_source"]["table_id"]
                if gold_table_id == gold_table_id:
                    is_table_hit = True 
                    if answer_text.lower() in block_content.lower():
                        is_block_hit = True
                        break
            if is_table_hit:
                table_hit.append(1)
            else:
                table_hit.append(0)
            
            if is_block_hit:
                block_hit.append(1)
            else:
                block_hit.append(0)
        
        logger.info(f"BM25 Retrieval Top-{top_k} Table Hit: {(sum(table_hit)/len(table_hit)) * 100}")
        logger.info(f"BM25 Retrieval Top-{top_k} Block Hit: {(sum(block_hit)/len(block_hit)) * 100}")


    """
    block_to_tableid = {}
    blocks = []
    for doc in tqdm(data, desc="Creating blocks..."):
        psg = doc['passages'][:8] # top 8 passages, following the encode_datasets.py for other models
        linearized_fused_block = linearize_block(doc['table'], psg, doc['meta_data'])
        block_to_tableid[linearized_fused_block] = {
            "table_id": doc["table_id"]
        }
        blocks.append(linearized_fused_block)
    
    logger.info(f"Loading OTT-QA dev data from {args.dev_path}")
    with open(args.dev_path, 'r') as f:
        dev_data = json.load(f)
    
    retriever = BM25Retriever(blocks)

    retriever.create_index()

    logger.info("Evaluating retriever...")
    evaluate_retriever(retriever, dev_data, block_to_tableid)
    logger.info("Retriever evaluated!")
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_path", type=str, default="/home/deokhk/research/OTT-QA-based-research/FR_with_rank/data_ottqa/dev.json", help="Path to OTT-QA dev data")
    parser.add_argument("--data_path", type=str, default="/home/deokhk/research/OTT-QA-based-research/FR_with_rank/preprocessed_data/retrieval/table_corpus_blink.pkl", help="Path to blink table corpus")

    args = parser.parse_args()
    main(args)