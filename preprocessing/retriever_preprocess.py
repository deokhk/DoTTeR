import argparse
import json
import pickle
import os
import sys 
import copy
import random
import logging
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import numpy as np

import nltk
import nltk.data

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from dateparser.search import search_dates
from dateparser import parse
from utils_preprocess import getrank_rowfield, read_json
from transformers import pipeline, AutoTokenizer
from rank_bm25 import BM25Okapi

sys.path.append('../')

from scripts.train_false_positive_removal import convert_datapoint_to_text


logger = logging.getLogger()
nltk.download('stopwords')

stopWords = set(stopwords.words('english'))
tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(2, 3), stop_words=stopWords)
# tfidf = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 3), stop_words=stopWords)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
basic_dir = '..'
resource_path = f'{basic_dir}/data_wikitable/'
best_threshold = 0.80


def get_sorted_sentences(tfidf, tfidf_query, all_index, all_passages):
    keys = []
    paras = []
    # threshold = 0.99
    for k, v in zip(all_index, all_passages):
        for _ in tokenizer.tokenize(v):
            keys.append(k)
            paras.append(_)
    try:
        para_feature = tfidf.fit_transform(paras)
        transformed = True
    except Exception:
        # logger.info("only containing stop words, skip it")
        transformed = False
    if transformed:
        q_feature = tfidf.transform(tfidf_query)
        para_tfidf_dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]
        sorted_passages = [(k, para, d) for k, para, d in zip(keys, paras, para_tfidf_dist)]
        sorted_passages = sorted(sorted_passages, key=lambda x: x[2], reverse=False)  # 升序
        return sorted_passages
    else:
        return [(k, para, 1) for k, para in zip(keys, paras)]


def get_sorted_passages(tfidf, tfidf_query, all_index, all_passages):
    keys = []
    paras = []
    # threshold = 0.99
    for k, v in zip(all_index, all_passages):
        keys.append(k)
        paras.append(v)
    try:
        para_feature = tfidf.fit_transform(paras)
        transformed = True
    except Exception:
        # logger.info("only containing stop words, skip it")
        transformed = False
    if transformed:
        q_feature = tfidf.transform(tfidf_query)
        para_tfidf_dist = pairwise_distances(q_feature, para_feature, 'cosine')[0]
        sorted_passages = [(k, para, d) for k, para, d in zip(keys, paras, para_tfidf_dist)]
        sorted_passages = sorted(sorted_passages, key=lambda x: x[2], reverse=False)  # 升序
        return sorted_passages
    else:
        return [(k, para, 1) for k, para in zip(keys, paras)]


def get_passages_traindev(tfidf_query, first_batch_passages, second_batch_passages, mode):
    fir_index, fir_passages = first_batch_passages['idx'], first_batch_passages['psg']  # often answer_node_passages
    sec_index, sec_passages = second_batch_passages['idx'], second_batch_passages['psg']  # often gpt_passages to append

    output_index = []
    output_passages = []
    if mode=='sort_sent':
        sorted_instances = get_sorted_sentences(tfidf, tfidf_query, fir_index, fir_passages)

        output_passages = [item[1] for item in sorted_instances]
        output_index = [item[0] for item in sorted_instances]

    elif mode=='sort_psg':
        sorted_instances = get_sorted_passages(tfidf, tfidf_query, fir_index, fir_passages)

        output_passages = [item[1] for item in sorted_instances]
        output_index = [item[0] for item in sorted_instances]

    elif mode=='append_sort_psg':
        # if len(sec_passages) > 0:
        sorted_instances = get_sorted_passages(tfidf, tfidf_query, sec_index, sec_passages)
        sorted_passages = [item[1] for item in sorted_instances]
        sorted_index = [item[0] for item in sorted_instances]
        appended_index = [num_id for num_id, idx in enumerate(sorted_index) if idx not in fir_index]

        output_index = fir_index + [sorted_index[num_id] for num_id in appended_index]
        output_passages = fir_passages + [sorted_passages[num_id] for num_id in appended_index]

    elif mode=='sort_append_sort_psg':
        # if len(sec_passages) > 0:
        sorted_instances = get_sorted_passages(tfidf, tfidf_query, fir_index, fir_passages)
        sorted_passages_fir = [item[1] for item in sorted_instances]
        sorted_index_fir = [item[0] for item in sorted_instances]

        sorted_instances = get_sorted_passages(tfidf, tfidf_query, sec_index, sec_passages)
        sorted_passages = [item[1] for item in sorted_instances]
        sorted_index = [item[0] for item in sorted_instances]
        appended_index = [num_id for num_id, idx in enumerate(sorted_index) if idx not in sorted_index_fir]

        output_index = sorted_index_fir + [sorted_index[num_id] for num_id in appended_index]
        output_passages = sorted_passages_fir + [sorted_passages[num_id] for num_id in appended_index]
    stats = {'num-fir': len(fir_index), 'num-sec': len(sec_index), 'num-out': len(output_index)}
    assert(len(output_index)==len(output_passages))
    return {'idx': output_index, 'psg': output_passages, 'stats': stats}


def prepare_training_data_intable(zipped_data, table_path, request_path, denoise_pipeline=None):
    # load data
    # Here, data is a single datapoint (in debugging mode.)
    # passage links is also a dictionary, whose keys are row ids, and values are a list of linked passages.
    data, passage_links = zipped_data[0], zipped_data[1]
    table_id = data['table_id']
    # Loading the table/request information
    with open(f'{resource_path}/{table_path}/{table_id}.json'.encode('utf8')) as f:
        table = json.load(f)
    with open(f'{resource_path}/{request_path}/{table_id}.json'.encode('utf8')) as f:
        requested_documents = json.load(f)
    linked_passage_dict = {}
    if len(passage_links) > 0:
        for key, value in passage_links.items():
            row_id = int(key.split('_')[-1])
            linked_passage_dict[row_id] = [[psg["id"] for psg in value],
                                           [psg["text"] for psg in value]]  # psg["title_text"], psg["query"]

    # process table
    header = [inst[0] for inst in table['header']]
    contents = [[cell[0] for cell in row] for row in table['data']]
    if len(' '.join(header).split(' ')) > 100:
        logger.info("HEA::{}::{}".format(len(' '.join(header).split(' ')), header))

    raw_tables = []
    for i, row in enumerate(contents):
        raw_df = pd.DataFrame(data=[row], columns=header)
        raw_df = raw_df.applymap(str)
        raw_tables.append(raw_df)

    meta_data = {'title': table['title'],
                 'section_title': table['section_title'],
                 'header': header}
    answer = data['answer-text']

    # Mapping entity link to cell, entity link to surface word. Assure the position of every cell containing the links
    mapping_entity = {}
    position2wiki = []
    for row_idx, row in enumerate(table['data']):
        position2wiki_i = {}
        for col_idx, cell in enumerate(row):
            for i, ent in enumerate(cell[1]):
                mapping_entity[ent] = mapping_entity.get(ent, []) + [(row_idx, col_idx)]
                position2wiki_i[f'{row_idx},{col_idx}'] = position2wiki_i.get(f'{row_idx},{col_idx}', []) + [ent]
        position2wiki.append(position2wiki_i)

    # Get the index of positive examples and negative examples from answer-node
    # train/dev instance contains answer-node and not None, 7657/41469 table containing answers in table also in passage
    positive_index = [node[1] for node in data['answer-node']]
    positive_row_index_in_answer_node = list(set([node[1][0] for node in data['answer-node']]))
    all_positive_row_index = copy.deepcopy(positive_row_index_in_answer_node)


    compare_output = []
    # Extract the passages for each line of the table, a list of str
    in_table_in_passage = []
    passages = []
    passages_index = []
    passages_node_index = []
    table_passages = []
    for _i, (x, y) in enumerate(positive_index):
        if data['answer-node'][_i][2]:
            table_passages.append({'index': data['answer-node'][_i][2],
                                   'position': [x, y],
                                   'passage': requested_documents[data['answer-node'][_i][2]]})

    for i, row in enumerate(contents):
        tfidf_query = [data['question'] + ' ' + ' '.join(header) + ' ' + ' '.join(row)]
        # import pdb;  pdb.set_trace()
        if i in positive_row_index_in_answer_node:
            _index_cell = list(set([data['answer-node'][_i][2] for _i, (x, y) in enumerate(positive_index) if x == i]))
            _index = list(set([item for sublist in position2wiki[i].values() for item in sublist]))
            if None in _index:
                _index.remove(None)
            raw_passages = [requested_documents[index] for index in _index] # all raw passage in current "row"


            if i in linked_passage_dict:
                gt_index, gt_passages = linked_passage_dict[i][0], linked_passage_dict[i][1]
                out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                                first_batch_passages={'idx': _index, 'psg': raw_passages},
                                                second_batch_passages={'idx': gt_index, 'psg': gt_passages},
                                                mode='append_sort_psg')
            else:
                # If this row does not contain any passage..
                gt_index, gt_passages = [], []
                out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                                first_batch_passages={'idx': _index, 'psg': raw_passages},
                                                second_batch_passages={'idx': [], 'psg': []},
                                                mode='sort_psg')

            passages.append(out_psg['psg'])
            passages_index.append({'all_index': _index + gt_index, 's_index': out_psg['idx']})
            passages_node_index.append([data['answer-node'][_i] for _i, (x, y) in enumerate(positive_index) if x == i])
            compare_output.append({'has': out_psg['stats'], 'no': {}})

            # check in table or in passage
            where_position = list(set([data['answer-node'][_i][3] for _i, (x, y) in enumerate(positive_index) if x == i]))
            in_table_in_passage.append(where_position)

        else:
            # if line i is not in answer node, then take all the passages in this line
            # then judge whether the answer is in passages. If so, add this line to all_positive_row_index
            _index = list(set([item for sublist in position2wiki[i].values() for item in sublist]))
            if None in _index:
                _index.remove(None)
            raw_passages = [requested_documents[index] for index in _index] 
            if i in linked_passage_dict:
                gt_index, gt_passages = linked_passage_dict[i][0], linked_passage_dict[i][1]
                out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                                first_batch_passages={'idx': gt_index, 'psg': gt_passages},
                                                second_batch_passages={'idx': _index, 'psg': raw_passages},
                                                mode='sort_append_sort_psg')
            else:
                gt_index, gt_passages = [], []
                out_psg = get_passages_traindev(tfidf_query=tfidf_query,
                                                first_batch_passages={'idx': _index, 'psg': raw_passages},
                                                second_batch_passages={'idx': [], 'psg': []},
                                                mode='sort_psg')
            output_passages = out_psg['psg']
            passages.append(output_passages)
            passages_index.append({'all_index': _index + gt_index, 's_index': out_psg['idx']})
            passages_node_index.append([data['answer-node'][_i] for _i, (x, y) in enumerate(positive_index) if x == i])
            compare_output.append({'has': {}, 'no': out_psg['stats']})
            in_table_in_passage.append([])

            # if answer in ' '.join(output_passages[:3]):  # TODO 有可能在取512个token时，就把含有answer的psg截断了，所以这个3不是那么合理
            #     all_positive_row_index.append(i)

    # For OTTeR_random

    if args.random_remain_k:
        # Sample one random row
        all_positive_row_index = random.sample(all_positive_row_index, min(len(all_positive_row_index), args.k))

    # Let's do denoising here!
    if len(all_positive_row_index) > 1 and denoise_pipeline is not None:
        # We denoise the all_positive_row_index.
        # For each possible positive row candidate, we check whether the row is related to the question.
        # Those with high probability of being related to the question are kept.
        question = data['question']
        row_text_list = []
        for index in all_positive_row_index:
            text = convert_datapoint_to_text(question, raw_tables[index], passages[index], meta_data)
            row_text_list.append(text)
        denoise_results = denoise_pipeline(row_text_list, padding=True, truncation=True)

        # Select those with highest score 
        best_score = -1
        best_index = all_positive_row_index[0]
        for index, result in zip(all_positive_row_index, denoise_results):
            if result['label'] == "POSITIVE" and (result['score'] > best_score):
                best_score = result['score']
                best_index = index
        
        all_positive_row_index = [best_index]

    # Merge a question instance to multiple positive and negative examples
    negative_row_index = [i for i in range(len(contents)) if i not in all_positive_row_index]
    # if len(negative_row_index) == 0:
    #     negative_row_index = random.sample(set(all_positive_row_index)-set(positive_row_index_in_answer_node), 1)
    #     logger.info("No negative rows. #rows:{}/{}".format(len(positive_row_index_in_answer_node), len(all_positive_row_index)))
    processed_data = []
    # import pdb;  pdb.set_trace()
    for i, row in enumerate(contents):
        example = {}
        if i in all_positive_row_index:
            example['question_id'] = data['question_id'] + f'-{len(processed_data)}'
            example['question'] = data['question']
            example['label'] = 1
            example['table_id'] = data['table_id']
            example['meta_data'] = meta_data
            example['answer-text'] = answer
            example['question_postag'] = data['question_postag']
            example['answer-node'] = passages_node_index[i]

            example['table'] = raw_tables[i]
            example['passages'] = passages[i]
            example['passages_id'] = passages_index[i]
            example['row_id'] = i

            # Calculate rank information 
            # TODO: We calculate the rank information by default. Need to make it controllable
            pos_rank_list = getrank_rowfield(i, contents, header)
            example['pos_sample_rank_list'] = pos_rank_list

            # process negatives
            if len(negative_row_index) > 1:
                neg_i = random.sample(negative_row_index, 1)[0]
                negative_row_index.remove(neg_i)
            elif len(negative_row_index) == 1:
                neg_i = negative_row_index[0]
            else:
                population = list(range(len(contents)))
                population.remove(i)
                neg_i = random.sample(population, 1)[0]

            if args.nega == 'intable_contra':
                # import pdb;  pdb.set_trace()
                if len(in_table_in_passage[i]) == 2:
                    example['neg_table'] = raw_tables[neg_i]
                    example['neg_passages'] = passages[neg_i]
                    example['neg_passages_id'] = passages_index[neg_i]
                    example['neg_row_id'] = neg_i
                elif len(in_table_in_passage[i])==1 and in_table_in_passage[i][0] == 'table':
                    example['neg_table'] = raw_tables[neg_i]
                    example['neg_passages'] = passages[i]
                    example['neg_passages_id'] = passages_index[i]
                    example['neg_row_id'] = neg_i
                elif len(in_table_in_passage[i])==1 and in_table_in_passage[i][0] == 'passage':
                    example['neg_table'] = raw_tables[i]
                    example['neg_passages'] = passages[neg_i]
                    example['neg_passages_id'] = passages_index[neg_i]
                    example['neg_row_id'] = i
                else:
                    print(in_table_in_passage)
                    continue
            elif args.nega == 'intable_random':
                example['neg_table'] = raw_tables[neg_i]
                example['neg_passages'] = passages[neg_i]
                example['neg_passages_id'] = passages_index[neg_i]
                example['neg_row_id'] = neg_i
            elif args.nega == 'intable_bm25':
                corpus = [' '.join(row)+' '.join(p) for row, p in zip(contents, passages)]
                tokenized_corpus = [doc.split(" ") for doc in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                query = data['question'].split(' ')
                doc_scores = bm25.get_scores(query)
                index = np.array([0 if i in all_positive_row_index else 1 for i in range(len(doc_scores))])
                neg_i = (doc_scores*index).argmax()
                example['neg_table'] = raw_tables[neg_i]
                example['neg_passages'] = passages[neg_i]
                example['neg_passages_id'] = passages_index[neg_i]
                example['neg_row_id'] = neg_i
            else:
                raise NotImplementedError

            # Append rank information of the negative example
            if example['neg_row_id'] == i:
                neg_ranK_list = getrank_rowfield(i, contents, header)
                example['neg_sample_rank_list'] = neg_ranK_list

            elif example["neg_row_id"] == neg_i: 
                neg_ranK_list = getrank_rowfield(neg_i, contents, header)
                example['neg_sample_rank_list'] = neg_ranK_list
            else:
                pass 
            processed_data.append(example)
    return {'retrieval_data': processed_data, 'compare_output': compare_output}


def get_filename(args):
    prefix = args.nega
    if args.aug_blink:
        prefix += '_blink'
    prefix += '_row'
    prefix += '_{}'.format(args.prefix) if args.prefix else ''

    if args.denoise:
        prefix += '_denoise'
    
    if args.random_remain_k:
        prefix += '_random_k{}'.format(args.k)
    save_path = '{}/preprocessed_data/retrieval/{}_{}.pkl'.format(basic_dir, args.split, prefix)
    cmp_save_path = '{}/preprocessed_data/retrieval/stats_{}_{}.json'.format(basic_dir, args.split, prefix)  # metagptdoc_woextraposi
    all_blocks_path = '{}/preprocessed_data/retrieval/{}ALLBLOCKS_{}.pkl'.format(basic_dir, args.split, prefix)
    logger.info(" save_path :{}".format(save_path))
    logger.info(" cmp_save_path :{}".format(cmp_save_path))
    logger.info(" all_blocks_path :{}".format(all_blocks_path))
    return save_path, cmp_save_path, all_blocks_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', type=str, choices=['train', 'dev', 'table_corpus', 'table_corpus_blink'])
    parser.add_argument('--model', default='../../OTT-QA/retriever/title_sectitle_schema/index-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz', type=str)
    parser.add_argument('--nega', default='intable_random', type=str, choices=['intable_random', 'intable_contra', 'intable_bm25'])
    parser.add_argument('--prefix', default='', type=str, help="['blink', 'metagptdoc_woextraposi', 'metagptdoc_append', 'metagptdoc_row', 'tb', 'double']")
    # parser.add_argument('--task', default='ottqa-train', type=str, choices=['ottqa-train', 'ottqa-dev', 'ottqa-test'])
    parser.add_argument('--max_passages', default=3, type=int)
    parser.add_argument('--max_tokens', default=360, type=int)
    parser.add_argument('--max_cell_tokens', default=36, type=int)
    parser.add_argument('--run_id', default=1, type=int)
    parser.add_argument('--reprocess', action='store_true')
    parser.add_argument('--aug_blink',action='store_true', help="augment retrieval data with blink links")

    # Denoising arguments 
    parser.add_argument('--denoise', action='store_true', help="Whether to denoise the dataset")
    parser.add_argument('--denoise_model_path', default=None, type=str, help="Path to the denoising model")

    # Experimental arguments
    parser.add_argument('--random_remain_k', action='store_true', help="For OTTeR_random and OTTeR_N experiment")
    parser.add_argument('--k', default=1, type=int, help="Randomly remain k rows in the example from same table")

    # parser.add_argument('--contra_double', action='store_true')
    # parser.add_argument('--positive_cell', action='store_true')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    basic_dir = '..'  
    n_threads = 24

    if args.denoise:
        logger.info("Denoising the dataset")
        assert args.denoise_model_path is not None
        pipeline_tokenizer = AutoTokenizer.from_pretrained(args.denoise_model_path, model_max_length=512)
        denoise_pipeline = pipeline("text-classification", model=args.denoise_model_path, tokenizer=pipeline_tokenizer, device=0)
        logger.info("Loading the denoising model from {}".format(args.denoise_model_path))        

    if args.split in ['train', 'dev']:
        logger.info("using {}".format(args.split))
        table_path = 'traindev_tables_tok'
        request_path = 'traindev_request_tok'
        save_path, cmp_save_path, all_blocks_path = get_filename(args)
        if not os.path.exists(save_path) or args.reprocess:
            with open('{}/data_ottqa/{}.traced.json'.format(basic_dir, args.split), 'r') as f:
                data = json.load(f)
            logger.info("num of data: {} ".format(len(data)))
            results = []

            table2tbid = {item['table_id']:{} for item in data}
            logger.info("num of table2tbid: {} ".format(len(table2tbid)))
            if args.aug_blink:
                path = f'{basic_dir}/data_wikitable/all_constructed_blink_tables.json'
                logger.info("aug_blink: ")
                logger.info("reading constructed tables with links from: {}".format(path))
                data_links = read_json(path)
                with open(f'{basic_dir}/data_wikitable/all_passages.json', 'r', encoding='utf-8') as fp:
                    all_passages = json.load(fp)
                for tbid in table2tbid.keys():
                    table = data_links.get(tbid, None)
                    if table:
                        if table2tbid.get(tbid, None):
                            for table_block_id, combined_psgs in table2tbid[tbid].items():
                                row_id = int(table_block_id.split('_')[-1])
                                old_psg_keys = [item['id'] for item in combined_psgs]
                                blink_ids = [psg_id for cell in table['data'][row_id] for psg_id in cell[1]]
                                plus_ids = [id for id in blink_ids if id not in old_psg_keys]
                                plus_data = [{'id': id, 'text': all_passages[id],
                                              'title_text': ' '.join(id[6:].split('_')),
                                              'query': ' '.join(id[6:].split('_'))} for id in plus_ids]
                                table2tbid[tbid][table_block_id].extend(plus_data) 
                        else:
                            for row_id, row in enumerate(table['data']):
                                table_block_id = '{}_{}'.format(tbid, row_id)
                                plus_ids = [psg_id for cell in row for psg_id in cell[1]]
                                plus_data = [{'id': id, 'text': all_passages[id],
                                              'title_text': ' '.join(id[6:].split('_')),
                                              'query': ' '.join(id[6:].split('_'))} for id in plus_ids] 
                                table2tbid[tbid][table_block_id] = plus_data

            logger.info("{}/{} tables have replaced linked passages".format(sum([len(tbs)>0 for tbs in table2tbid.values()]), len(table2tbid)))

            # import pdb;  pdb.set_trace()
            zipped_data = [(item, table2tbid[item['table_id']]) for item in data]

            
            if not args.denoise:
                with Pool(n_threads) as p:
                    func_ = partial(prepare_training_data_intable, table_path=table_path, request_path=request_path,)
                    all_results = list(tqdm(p.imap(func_, zipped_data, chunksize=16), total=len(data), desc="convert examples to trainable data",))
                    results = [res['retrieval_data'] for res in all_results]
                    compare_output = [res['compare_output'] for res in all_results]
            else:
                # denoise 
                all_data = []
                for d in tqdm(zipped_data, desc="Convert examples to trainable data (with denoising)"):
                    data = prepare_training_data_intable(d, table_path=table_path, request_path=request_path, denoise_pipeline=denoise_pipeline)
                    all_data.append(data)
                results = [data['retrieval_data'] for data in all_data]
                compare_output = [data['compare_output'] for data in all_data]

            results = [item for inst in results for item in inst]
            random.shuffle(results)
            logger.info(f"Number of datapoints: {len(results)}")
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            with open(cmp_save_path, 'w', encoding='utf-8') as f:
                json.dump(compare_output, f, indent=4)

    else:
        raise NotImplementedError