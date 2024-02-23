# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the 
# LICENSE file in the root directory of this source tree.
import csv
import json
# import pickle
import pandas as pd
import _pickle as pickle
# import _pickle as cPickle
# import cPickle as pickle

import logging
import pdb
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import codecs
import unicodedata
import re
import os

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



from .data_utils import collate_tokens, collate_tokens_to_3d
# from .tr_dataset import convert_tb_to_features
from .tr_dataset import convert_tb_to_features_bert, convert_tb_to_features_bert_metadata
from .tr_dataset import convert_tb_to_features_bert_metadata_threecat
from .tr_dataset import convert_tb_to_features_bert_metadata_threecat_one_query
from .tr_dataset import convert_tb_to_features_tapas, convert_tb_to_features_tapas_metadata
from .tr_dataset import convert_tb_to_features_bert_metadata_threecat_with_rank_one_query, convert_tb_to_features_bert_metadata_threecat_with_rank
from .tr_dataset import convert_tb_to_features_bert_metadata_threecat_with_global_one_query
from .tr_dataset import convert_tb_to_features_bert_metadata_threecat_with_column_global_one_query
from .tr_dataset import convert_tb_to_features_bert_metadata_threecat_with_column_fusion_one_query
from .tr_dataset import get_passages

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string

class EmDataset(Dataset):

    def __init__(self, tokenizer, data_path, table, args):
        super(EmDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.table = table  # Bool, indicate encoding table block (True) or query (False)
        self.args = args
        self.get_item = None
        self.save_path = ""
        self.data = []
        self.get_item = self.get_item_tapas if 'tapas' in self.args.model_name else self.get_item_bert
        self.function = convert_tb_to_features_tapas if 'tapas' in self.args.model_name else convert_tb_to_features_bert

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        if self.table:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        # self.data = self.data[177600:]
        logger.info(f"Total sample count {len(self.data)}")

        if self.table:
            if not os.path.exists(self.args.embed_save_path):
                os.makedirs(self.args.embed_save_path)
            self.save_path = os.path.join(self.args.embed_save_path, "id2doc.json")  # ID to doc mapping
            if not os.path.exists(self.save_path):
                id2doc = {}
                for idx, doc in enumerate(self.data):
                    id2doc[idx] = {"table_id": doc["table_id"],
                                   "row_id": doc['row_id'],
                                   "table": [doc['table'].columns.tolist(), doc['table'].values.tolist()],
                                   "meta_data": doc.get('meta_data', None),
                                   "passages_id": doc['passages_id'],
                                   "passages": doc['passages'],}
                with open(self.save_path, "w") as g:
                    json.dump(id2doc, g, indent=1)
                logger.info(f"Dumping id2doc to {self.save_path}")

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.data)

    def get_item_tapas(self, index):
        js = self.data[index]
        if self.table:
            table_block_tensor = self.function(' '.join(js['passages']), js['table'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }

    def get_item_bert(self, index):
        js = self.data[index]

        if self.table:
            table_block_tensor = self.function(' '.join(js['passages']), js['table'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }


class EmDatasetMeta(EmDataset):

    def __init__(self, tokenizer, data_path, table, args):
        super(EmDatasetMeta, self).__init__(tokenizer, data_path, table, args)
        self.function = convert_tb_to_features_tapas_metadata if 'tapas' in self.args.model_name else convert_tb_to_features_bert_metadata

    def get_item_tapas(self, index):

        js = self.data[index]
        if self.table:
            psg = ' '.join(get_passages(js, self.args.psg_mode, neg=False))[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), meta_data=None, tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }

    def get_item_bert(self, index):
        js = self.data[index]

        if self.table:
            psg = get_passages(js, self.args.psg_mode, neg=False)[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), meta_data=None, tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }


class EmDatasetMetaThreeCat(EmDataset):

    def __init__(self, tokenizer, data_path, table, args):
        super(EmDatasetMetaThreeCat, self).__init__(tokenizer, data_path, table, args)
        self.function = convert_tb_to_features_tapas_metadata if 'tapas' in self.args.model_name else convert_tb_to_features_bert_metadata_threecat
        if args.one_query:
            self.function = convert_tb_to_features_bert_metadata_threecat_one_query

    def get_item_tapas(self, index):
        raise Exception("not implement three cat in emdataset")


    def get_item_bert(self, index):
        js = self.data[index]

        if self.table:
            if self.args.no_passages:
                psg = [' ', ' ']
            else:
                psg = get_passages(js, self.args.psg_mode, neg=False)[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), meta_data=None, tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }


class EmDatasetMetaRankThreeCat(EmDataset):
    def __init__(self, tokenizer, data_path, table, args):
        super(EmDatasetMetaRankThreeCat, self).__init__(tokenizer, data_path, table, args)
        if 'tapas' in args.model_name:
            raise Exception ('did not implement tapas with threecat setting')

        self.function = convert_tb_to_features_bert_metadata_threecat_with_rank
        if args.one_query:
            self.function = convert_tb_to_features_bert_metadata_threecat_with_rank_one_query

    def get_item_tapas(self, index):
        raise Exception("not implement three cat in emdataset")


    def get_item_bert(self, index):
        js = self.data[index]
        if self.table:
            if self.args.no_passages:
                psg = [' ', ' ']
            else:
                psg = get_passages(js, self.args.psg_mode, neg=False)[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], js['rank_list'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), meta_data=None, rank_info=None, tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }

class EmDatasetMetaGlobalThreeCat(EmDataset):

    def __init__(self, tokenizer, data_path, table, args):
        super(EmDatasetMetaGlobalThreeCat, self).__init__(tokenizer, data_path, table, args)
        if 'tapas' in args.model_name:
            raise Exception ('did not implement tapas with threecat setting')

        if args.one_query:
            self.function = convert_tb_to_features_bert_metadata_threecat_with_global_one_query
        else:
            raise NotImplementedError("Not implemented multi query for global three cat")

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        if self.table:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(f"Total sample count {len(self.data)}")


        if self.table:
            logger.info(f"Loading blink tables from {self.args.all_blink_table_path}")
            with open(self.args.all_blink_table_path, 'r') as f: 
                self.all_tables = json.load(f)
            # Summary token selection
            if "roberta" in self.args.model_name:
                self.summary_token = "madeupword0000"
            elif "bert" in self.args.model_name: 
                self.summary_token = "[unused0]"
            else:
                raise ValueError("Model not supported for summary injection")


        if self.table:
            if not os.path.exists(self.args.embed_save_path):
                os.makedirs(self.args.embed_save_path)
            self.save_path = os.path.join(self.args.embed_save_path, "id2doc.json")  # ID to doc mapping
            if not os.path.exists(self.save_path):
                id2doc = {}
                for idx, doc in enumerate(self.data):
                    id2doc[idx] = {"table_id": doc["table_id"],
                                   "row_id": doc['row_id'],
                                   "table": [doc['table'].columns.tolist(), doc['table'].values.tolist()],
                                   "meta_data": doc.get('meta_data', None),
                                   "passages_id": doc['passages_id'],
                                   "passages": doc['passages'],}
                with open(self.save_path, "w") as g:
                    json.dump(id2doc, g, indent=1)
                logger.info(f"Dumping id2doc to {self.save_path}")

    def get_item_tapas(self, index):
        raise Exception("not implement three cat in emdataset")


    def get_item_bert(self, index):
        js = self.data[index]
        if self.table and (js["table_id"] in self.all_tables):
            whole_table = self.all_tables[js["table_id"]]
        else:
            whole_table = None 

        if self.table:
            if self.args.no_passages:
                psg = [' ', ' ']
            else:
                psg = get_passages(js, self.args.psg_mode, neg=False)[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], summary_token=self.summary_token, whole_table=whole_table, tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), meta_data=None, summary_token=None, whole_table=None, tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }


class EmDatasetMetaColumnGlobalThreeCat(EmDataset):

    def __init__(self, column_model_tokenizer, tokenizer, data_path, table, args):
        super(EmDatasetMetaColumnGlobalThreeCat, self).__init__(tokenizer, data_path, table, args)
        self.column_model_tokenizer = column_model_tokenizer
        
        if 'tapas' in args.model_name:
            raise Exception ('did not implement tapas with threecat setting')

        if args.one_query:
            self.function = convert_tb_to_features_bert_metadata_threecat_with_column_fusion_one_query
        else:
            raise NotImplementedError("Not implemented multi query for column global three cat")

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        if self.table:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(f"Total sample count: {len(self.data)}")

        if self.table:
            logger.info(f"Loading blink tables from {self.args.all_blink_table_path}")
            with open(self.args.all_blink_table_path, 'r') as f: 
                self.all_tables = json.load(f)


        if self.table:
            if not os.path.exists(self.args.embed_save_path):
                os.makedirs(self.args.embed_save_path)
            self.save_path = os.path.join(self.args.embed_save_path, "id2doc.json")  # ID to doc mapping
            if not os.path.exists(self.save_path):
                id2doc = {}
                for idx, doc in enumerate(self.data):
                    id2doc[idx] = {"table_id": doc["table_id"],
                                   "row_id": doc['row_id'],
                                   "table": [doc['table'].columns.tolist(), doc['table'].values.tolist()],
                                   "meta_data": doc.get('meta_data', None),
                                   "passages_id": doc['passages_id'],
                                   "passages": doc['passages'],}
                with open(self.save_path, "w") as g:
                    json.dump(id2doc, g, indent=1)
                logger.info(f"Dumping id2doc to {self.save_path}")

    def get_item_tapas(self, index):
        raise Exception("not implement three cat in emdataset")


    def get_item_bert(self, index):
        js = self.data[index]
        if self.table and (js["table_id"] in self.all_tables):
            whole_table = self.all_tables[js["table_id"]]
        else:
            whole_table = None 

        if self.table:
            if self.args.no_passages:
                psg = [' ', ' ']
            else:
                psg = get_passages(js, self.args.psg_mode, neg=False)[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], whole_table, tokenizer=self.tokenizer, column_model_tokenizer=self.column_model_tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), None, None, tokenizer=self.tokenizer, column_model_tokenizer=self.column_model_tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }



class EmDatasetMetaFusionThreeCat(EmDataset):

    def __init__(self, column_model_tokenizer, tokenizer, data_path, table, args):
        super(EmDatasetMetaFusionThreeCat, self).__init__(tokenizer, data_path, table, args)
        self.column_model_tokenizer = column_model_tokenizer
        
        if 'tapas' in args.model_name:
            raise Exception ('did not implement tapas with threecat setting')

        if args.one_query:
            self.function = convert_tb_to_features_bert_metadata_threecat_with_column_fusion_one_query
        else:
            raise NotImplementedError("Not implemented multi query for column global three cat")

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        if self.table:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        logger.info(f"Total sample count: {len(self.data)}")

        if self.table:
            logger.info(f"Loading blink tables from {self.args.all_blink_table_path}")
            with open(self.args.all_blink_table_path, 'r') as f: 
                self.all_tables = json.load(f)
            # Summary token selection
            if "roberta" in self.args.model_name:
                self.column_summary_token = "madeupword0000"
            elif "bert" in self.args.model_name: 
                self.column_summary_token = "[unused0]"
            else:
                raise ValueError("Model not supported for summary injection")


        if self.table:
            if not os.path.exists(self.args.embed_save_path):
                os.makedirs(self.args.embed_save_path)
            self.save_path = os.path.join(self.args.embed_save_path, "id2doc.json")  # ID to doc mapping
            if not os.path.exists(self.save_path):
                id2doc = {}
                for idx, doc in enumerate(self.data):
                    id2doc[idx] = {"table_id": doc["table_id"],
                                   "row_id": doc['row_id'],
                                   "table": [doc['table'].columns.tolist(), doc['table'].values.tolist()],
                                   "meta_data": doc.get('meta_data', None),
                                   "passages_id": doc['passages_id'],
                                   "passages": doc['passages'],}
                with open(self.save_path, "w") as g:
                    json.dump(id2doc, g, indent=1)
                logger.info(f"Dumping id2doc to {self.save_path}")

    def get_item_tapas(self, index):
        raise Exception("not implement three cat in emdataset")


    def get_item_bert(self, index):
        js = self.data[index]
        if self.table and (js["table_id"] in self.all_tables):
            whole_table = self.all_tables[js["table_id"]]
        else:
            whole_table = None 

        if self.table:
            if self.args.no_passages:
                psg = [' ', ' ']
            else:
                psg = get_passages(js, self.args.psg_mode, neg=False)[:8]
            table_block_tensor = self.function(psg, js['table'], js['meta_data'], whole_table, tokenizer=self.tokenizer, column_model_tokenizer=self.column_model_tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = self.function(question, pd.DataFrame([]), None, None, tokenizer=self.tokenizer, column_model_tokenizer=self.column_model_tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }





class EmDatasetFilter(EmDataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 tfidf_doc_path,
                 table,
                 args,
                 ):
            super().__init__()
            self.tokenizer = tokenizer
            self.table = table  # Bool, indicate encoding table block (True) or query (False)
            self.args = args
            # self.question =[]
            # self.table_block = []
            self.save_path = os.path.join(args.embed_save_path, "id2doc.json")
            logger.info(f"Loading data from {data_path}")
            self.data = []
            if not os.path.exists(args.embed_save_path):
                os.makedirs(args.embed_save_path)
            if self.table and os.path.exists(args.tmp_data_save_path):
                print(f"Loading data from {args.tmp_data_save_path}")
                with open(args.tmp_data_save_path,'rb') as f:
                    self.data = pickle.load(f)
                print('The length of the data is {}'.format(len(self.data)))
                return

            if self.table:
                with open(data_path, 'rb') as f:
                    self.raw_data = pickle.load(f)
            else:
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.raw_data = json.load(f)
            with open(tfidf_doc_path, 'r', encoding='utf8') as f:
                self.tfidf_qid2docs = json.load(f)
            self.all_required_tableids = []
            for k, v in self.tfidf_qid2docs.items():
                self.all_required_tableids+=v['doc_ids'][:self.args.top_k]
            # [v for k, v in self.tfidf_qid2docs.items() for item in v['doc_ids'][:self.args.top_k]]
            # self.data = self.data[177600:]
            # self.data = self.data[379200:]
            logger.info(f"Total sample count {len(self.data)}")

            logger.info('{}'.format(self.all_required_tableids[:5]))
            logger.info('{}'.format(len(self.all_required_tableids)))
              # ID to doc mapping

            if self.table:
                for idx, doc in enumerate(self.raw_data):
                    if doc['table_id'] in self.all_required_tableids:
                        self.data.append(doc)
                with open(self.args.tmp_data_save_path,'wb') as g:
                    print('The length of data is {}'.format(len(self.data)))
                    pickle.dump(self.data,g)
                id2doc = {}
                for idx, doc in enumerate(self.data):
                    id2doc[idx] = {"table_id": doc["table_id"],
                                   "row_id": doc['row_id'],
                                   "table": [doc['table'].columns.tolist(), doc['table'].values.tolist()],
                                   "meta_data": doc['meta_data'],
                                   "passages_id": doc['passages_id'],
                                   "passages": doc['passages'],}
                with open(self.save_path, "w") as g:
                    json.dump(id2doc, g, indent=1)

    def processing_data(self):
        pass

    def __getitem__(self, index):
        return self.get_item_bert(index)

    def __len__(self):
        return len(self.data)

    def get_item_tapas(self, index):
        js = self.data[index]
        if self.table:
            psg = ' '.join(get_passages(js, self.args.psg_mode, neg=False))[:8]
            table_block_tensor = convert_tb_to_features_tapas_metadata(psg, js['table'], js['meta_data'], tokenizer=self.tokenizer, args=self.args)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = convert_tb_to_features_tapas_metadata(question, pd.DataFrame([]), js['meta_data'], tokenizer=self.tokenizer, args=self.args)
        return {
                "tb": table_block_tensor,
                }

    def get_item_bert(self, index):
        js = self.data[index]

        if self.table:
            psg = ' '.join(get_passages(js, self.args.psg_mode, neg=False))[:8]
            table_block_tensor = convert_tb_to_features_bert_metadata(psg, js['table'],js['meta_data'],tokenizer=self.tokenizer, args=self.args,encode=True)
        else:
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            table_block_tensor = convert_tb_to_features_bert_metadata(question, pd.DataFrame([]),js['meta_data'], tokenizer=self.tokenizer, args=self.args,encode=True)
        return {
                "tb": table_block_tensor,
                }


def em_collate_tapas(samples, pad_id=0):
    if len(samples) == 0:
        return {}
    # logger.info(samples[0]["tb"]["input_ids"].shape)
    # logger.info(samples[0]["tb"]["token_type_ids"].shape)
    # logger.info(samples[0]["tb"]["attention_mask"].shape)
    batch = {
        'input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'input_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["tb"]:
        batch.update({
            'part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["tb"]:
        batch.update({
            'part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["tb"]:
        batch.update({
            'input_type_ids': collate_tokens([s["tb"]["token_type_ids"] for s in samples], 0,'2d')  # # token_type_ids[1, 512, 7]
            # 'input_type_ids': collate_tokens([s["tb"]["token_type_ids"].view(-1) for s in samples], 0)  # token_type_ids[1, 512]
        })

    return batch


def em_collate_bert(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'input_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["tb"]:
        batch.update({
            'part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["tb"]:
        batch.update({
            'part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["tb"]:
        batch.update({
            'input_type_ids': collate_tokens([s["tb"]["token_type_ids"].view(-1) for s in samples], 0)
        })

    return batch

def em_collate_bert_with_global(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'input_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["tb"]:
        batch.update({
            'part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["tb"]:
        batch.update({
            'part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["tb"]:
        batch.update({
            'input_type_ids': collate_tokens([s["tb"]["token_type_ids"].view(-1) for s in samples], 0)
        })

    if "table_input_ids" in samples[0]["tb"]:
        batch.update({
            'table_input_ids': collate_tokens([s["tb"]["table_input_ids"].view(-1) for s in samples], pad_id),
            'table_mask': collate_tokens([s["tb"]["table_mask"].view(-1) for s in samples], 0),
        })
        
    if "table_token_type_ids" in samples[0]["tb"]:
        batch.update({
            'table_type_ids': collate_tokens([s["tb"]["table_token_type_ids"].view(-1) for s in samples], 0)
        })
    
    if "column_input_ids_list" in samples[0]["tb"]:
        batch.update({
            'column_input_ids_list': collate_tokens_to_3d([s["tb"]["column_input_ids_list"] for s in samples], -1),
            'column_mask_list': collate_tokens_to_3d([s["tb"]["column_attention_mask_list"] for s in samples], -1),
            'column_token_indices_list': collate_tokens([s["tb"]["column_token_indices"].view(-1) for s in samples], -1),
        })
    
    if "value_mask" in samples[0]["tb"]:
        batch.update({
            "value_mask": collate_tokens([s["tb"]["value_mask"].view(-1) for s in samples], 0),
        })
    
    if "column_categories" in samples[0]["tb"]:
        batch.update({
            'column_categories': collate_tokens([s["tb"]["column_categories"].view(-1) for s in samples], -1),
        })

    return batch



