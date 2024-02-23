import json
import pickle
import os, sys
import random
from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz 
from .data_utils import collate_tokens, collate_tokens_to_3d
import copy
import sys
sys.path.append('../')
from preprocessing.utils_preprocess import extract_value_and_type
from utils.common import (
    convert_tb_to_string, 
    convert_tb_to_string_norm,
    convert_tb_to_string_metadata,
    convert_tb_to_string_metadata_norm,
    convert_tb_to_string_metadata_with_summary_token,
    convert_tb_to_string_metadata_norm_with_summary_token,
    convert_tb_to_string_metadata_norm_with_rank,
    convert_tb_to_string_metadata_with_rank,
    convert_whole_tb_to_string_norm,
    convert_whole_tb_to_string,
    construct_clsg_input_roberta,
    convert_tb_to_string_metadata_norm_with_column_level_summary_token,
    convert_whole_tb_to_normalized_columns,
    get_passages, load_jsonl)

from transformers import TapasTokenizer
# from transformers.models.tapas.tokenization_tapas import BatchEncoding
from transformers import BatchEncoding
from transformers.models.tapas.tokenization_tapas import add_numeric_table_values, add_numeric_values_to_question

from functools import partial
# from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import Pool
import logging

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_tb_to_features_bert_metadata(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    # inputs['length'] = len(inputs['input_ids'])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_threecat(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, text_pair=passage_str, max_length=args.max_q_len*2,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        
    cls_id = tokenizer.cls_token_id 
    eos_id = tokenizer.eos_token_id 
    zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

    inputs['part2_mask'] = torch.tensor([[0] +
                                         [1] * (zero_positions[1] - 1) +
                                         [0] * (inputs['length'] - zero_positions[1])])
    inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                         [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                         [0] * (inputs['length'] - zero_positions[3])])
    del inputs['special_tokens_mask']
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_threecat_one_query(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        
        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = copy.deepcopy(inputs['part2_mask'])
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)

        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                             [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                             [0] * (inputs['length'] - zero_positions[3])])

    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_threecat_with_rank_one_query(passages, table, meta_data, rank_info, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]
        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = copy.deepcopy(inputs['part2_mask'])
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm_with_rank(table, passages, meta_data, rank_info, args.rank_scheme)
        else:
            table_str, passage_str = convert_tb_to_string_metadata_with_rank(table, passages, meta_data, rank_info, args.rank_scheme)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        zero_positions = torch.where(inputs['special_tokens_mask'] == 1)[1]
        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                             [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                             [0] * (inputs['length'] - zero_positions[3])])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_threecat_with_global_one_query(passages, table_row, meta_data, summary_token, whole_table, tokenizer, args, encode=False):
    if table_row.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table_row, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        
        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = copy.deepcopy(inputs['part2_mask'])
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm_with_summary_token(table_row, passages, meta_data, summary_token)
        else:
            table_str, passage_str = convert_tb_to_string_metadata_with_summary_token(table_row, passages, meta_data, summary_token)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)

        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                             [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                             [0] * (inputs['length'] - zero_positions[3])])

    if whole_table:
        if args.normalize_summary_table:
            whole_table_str = convert_whole_tb_to_string_norm(whole_table)
        else:
            whole_table_str = convert_whole_tb_to_string(whole_table)
        table_inputs = tokenizer.encode_plus(whole_table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        
        inputs["table_input_ids"] = table_inputs["input_ids"]
        inputs["table_mask"] = table_inputs["attention_mask"]
        if "token_type_ids" in table_inputs:
            inputs["table_token_type_ids"] = table_inputs["token_type_ids"]

    else:
        # Whole table is None..
        inputs["table_input_ids"] = inputs["input_ids"]
        inputs["table_mask"] = inputs["attention_mask"]
        if "token_type_ids" in inputs:
            inputs["table_token_type_ids"] = inputs["token_type_ids"]

    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs



def convert_tb_to_features_bert_metadata_threecat_with_column_global_one_query(passages, table_row, meta_data, summary_token, whole_table, tokenizer, column_model_tokenizer, args, encode=False):
    if table_row.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table_row, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        
        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = copy.deepcopy(inputs['part2_mask'])
    else:
        assert args.normalize_table == True 
        table_str, passage_str = convert_tb_to_string_metadata_norm_with_column_level_summary_token(table_row, passages, meta_data, summary_token)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)

        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                             [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                             [0] * (inputs['length'] - zero_positions[3])])

    # We need column-by-column input for sumary generation model  
    assert args.normalize_summary_table == True


    if whole_table:
        # passage 
        column_level_inputs = convert_whole_tb_to_normalized_columns(whole_table)
        tokenized_columns = column_model_tokenizer(column_level_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=512)   
        csep_index = len(column_model_tokenizer)-1 # We assume that the last token is [C_SEP]
        column_token_indices = get_column_token_indices(csep_index, table_row, whole_table, tokenized_columns) # To be used for column-level summary generation
        inputs["column_input_ids_list"] = tokenized_columns["input_ids"]
        inputs["column_attention_mask_list"] = tokenized_columns["attention_mask"]
        inputs["column_token_indices"] = column_token_indices 
        if "token_type_ids" in tokenized_columns:
            inputs["column_token_type_ids_list"] = tokenized_columns["token_type_ids"]
    else:
        if not table_row.empty:
            # There's a case where we don't find corresponding whole table for the table row (very rarely)
            # In this case, we just use the table row as the whole table
            title = meta_data["title"]
            section_title = meta_data["section_title"]
            headers = table_row.columns.tolist()
            aggregated_columns = [[] for i in range(len(headers))]
            row_values = table_row.values.tolist()[0]
            for j, value in enumerate(row_values):
                aggregated_columns[j].append(value)

            column_level_inputs = []
            for header, aggregated_column in zip(headers, aggregated_columns):
                column_level_input = construct_clsg_input_roberta(header, aggregated_column, title, section_title)
                column_level_inputs.append(column_level_input)
            
            # Do the last of the process
            tokenized_columns = column_model_tokenizer(column_level_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=512)   
            csep_index = len(column_model_tokenizer)-1 # We assume that the last token is [C_SEP]
            tokenized_columns_ids = tokenized_columns["input_ids"]
            column_token_indices = []
            for tokenized_cid in tokenized_columns_ids:
                column_token_indices.append(torch.where(tokenized_cid ==50265)[0])
            column_token_indices = torch.stack(column_token_indices).squeeze()
            # 이 부분 고치기..
            inputs["column_input_ids_list"] = tokenized_columns["input_ids"]
            inputs["column_attention_mask_list"] = tokenized_columns["attention_mask"]
            inputs["column_token_indices"] = column_token_indices 
            if "token_type_ids" in tokenized_columns:
                inputs["column_token_type_ids_list"] = tokenized_columns["token_type_ids"]


    # batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    del inputs["special_tokens_mask"]

    return inputs


def convert_tb_to_features_bert_metadata_threecat_with_column_fusion_one_query(passages, table_row, meta_data, whole_table, tokenizer, column_model_tokenizer, args, encode=False):
    if table_row.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table_row, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
        
        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = copy.deepcopy(inputs['part2_mask'])
    else:
        assert args.normalize_table == True 
        table_str, passage_str = convert_tb_to_string_metadata_norm(table_row, passages, meta_data, args.max_c_len) # We don't use summary token here
        # We need to return the mask (1 if token corresponds to column 1, 2 if token corresponds to column 2, etc. 0 for other tokens)
        
        inputs = tokenizer(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_offsets_mapping=True, return_length=True)


        cls_id = tokenizer.cls_token_id 
        eos_id = tokenizer.eos_token_id 
        
        zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

        inputs['part2_mask'] = torch.tensor([[0] +
                                             [1] * (zero_positions[1] - 1) +
                                             [0] * (inputs['length'] - zero_positions[1])])
        inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                             [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                             [0] * (inputs['length'] - zero_positions[3])])


        # We need to retrieve entity span 
        if "fusion" in args.injection_scheme:
            values = table_row.values.tolist()[0]
            # Initialize a list to store entity offsets

            value_offsets = []
            tokens = inputs["input_ids"][0].tolist()
            # Construct value offset
            for value in values:
                tmp = f"is {value}"
                start_idx = table_str.find(tmp)
                start_idx = start_idx+3
                end_idx = start_idx + len(value)
                value_span = (start_idx, end_idx)
                value_offsets.append(value_span)

            assert len(value_offsets) == len(values), "Length of value offsets and values are not the same"

            # For each token, if the corresponding offset is within the value offset, we assign the token to the value
            value_mask = [0] * len(tokens)

            offset_mappings = inputs["offset_mapping"][0].tolist()
            for i, offset in enumerate(offset_mappings):
                if i!=0 and offset == [0, 0]:
                    # We reach the end of table string
                    break 
                for value_idx, value_offset in enumerate(value_offsets):
                    if value_offset[0] <= offset[0] and offset[1] <= value_offset[1]:
                        value_mask[i] = value_idx+1
                        break
            
            if args.injection_scheme == "fusion_first":
                # We only keep the first token corresponding to the value
                satisfied = set()
                for idx, v in enumerate(value_mask):
                    if v != 0:
                        if v not in satisfied:
                            satisfied.add(v)
                        else:
                            value_mask[idx] = 0
            inputs["value_mask"] = torch.tensor([value_mask])



    # We need column-by-column input for sumary generation model  
    assert args.normalize_summary_table == True


    if whole_table:
        # passage 
        column_level_inputs = convert_whole_tb_to_normalized_columns(whole_table)
        tokenized_columns = column_model_tokenizer(column_level_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=512)   
        csep_index = len(column_model_tokenizer)-1 # We assume that the last token is [C_SEP]
        column_token_indices = get_column_token_indices(csep_index, table_row, whole_table, tokenized_columns) # To be used for column-level summary generation
        inputs["column_input_ids_list"] = tokenized_columns["input_ids"]
        inputs["column_attention_mask_list"] = tokenized_columns["attention_mask"]
        inputs["column_token_indices"] = column_token_indices 
        if "token_type_ids" in tokenized_columns:
            inputs["column_token_type_ids_list"] = tokenized_columns["token_type_ids"]
        
        # column category
        column_categories = [] # 1 if numeric or date, 0 otherwise

        headers = [h[0] for h in whole_table["header"]]  
        rows = whole_table["data"]

        aggregated_columns = [[] for i in range(len(headers))]
        for r in rows:
            for j, value in enumerate(r):
                cell_value = value[0]
                aggregated_columns[j].append(cell_value)

        for header, aggregated_column in zip(headers, aggregated_columns):
            _, _, ctype = extract_value_and_type(aggregated_column)
            if ctype == "numeric" or ctype == "date":
                column_categories.append(1)
            else:
                column_categories.append(0)

        inputs["column_categories"] = torch.tensor([column_categories])

    else:
        if not table_row.empty:
            # There's a case where we don't find corresponding whole table for the table row (very rarely)
            # In this case, we just use the table row as the whole table
            title = meta_data["title"]
            section_title = meta_data["section_title"]
            headers = table_row.columns.tolist()
            aggregated_columns = [[] for i in range(len(headers))]
            row_values = table_row.values.tolist()[0]
            for j, value in enumerate(row_values):
                aggregated_columns[j].append(value)

            column_level_inputs = []
            column_categories = [] # 1 if numeric or date, 0 otherwise
            for header, aggregated_column in zip(headers, aggregated_columns):
                _, _, ctype = extract_value_and_type(aggregated_column)
                if ctype == "numeric" or ctype == "date":
                    column_categories.append(1)
                else:
                    column_categories.append(0)
                column_level_input = construct_clsg_input_roberta(header, aggregated_column, title, section_title)
                column_level_inputs.append(column_level_input)
            
            # Do the last of the process
            tokenized_columns = column_model_tokenizer(column_level_inputs, return_tensors="pt", padding="max_length", truncation=True, max_length=512)   
            csep_index = len(column_model_tokenizer)-1 # We assume that the last token is [C_SEP]
            tokenized_columns_ids = tokenized_columns["input_ids"]
            column_token_indices = []
            for tokenized_cid in tokenized_columns_ids:
                column_token_indices.append(torch.where(tokenized_cid ==50265)[0])
            column_token_indices = torch.stack(column_token_indices).squeeze()
            # 이 부분 고치기..
            inputs["column_input_ids_list"] = tokenized_columns["input_ids"]
            inputs["column_attention_mask_list"] = tokenized_columns["attention_mask"]
            inputs["column_token_indices"] = column_token_indices 
            inputs["column_categories"] = torch.tensor([column_categories])
            if "token_type_ids" in tokenized_columns:
                inputs["column_token_type_ids_list"] = tokenized_columns["token_type_ids"]


    # batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    del inputs["special_tokens_mask"]

    return inputs





def get_column_token_indices(csep_index, table_row, whole_table, tokenized_columns):
    """
    Get the [C_SEP] token indices of each column in the table, corresponding to the current table_row.
    """
    # First get the row index in the whole table 
    row_idx = -1 
    table_row_values = table_row.values.tolist()[0]
    table_row_values_stripped = [x.strip() for x in table_row_values]
    table_rows = whole_table["data"]
    for idx, row in enumerate(table_rows):
        row_values = [x[0].strip() for x in row]
        if row_values == table_row_values_stripped:
            row_idx = idx  
    if row_idx == -1:
        # Sometimes, table row is slightly different from the whole table or not found in the whole table
        # We perform fuzzy string matching to find the row index
        table_row_values_stripped_str = " ".join(table_row_values_stripped)
        maximum_similarity = -1
        maximum_idx = -1
        for idx, row in enumerate(table_rows):
            row_values = [x[0].strip() for x in row]
            row_values_str = " ".join(row_values)
            similarity = fuzz.ratio(table_row_values_stripped_str, row_values_str)
            if similarity > maximum_similarity:
                maximum_similarity = similarity
                maximum_idx = idx
        row_idx = maximum_idx

    # Then get the column indices
    tokenized_columns_ids = tokenized_columns["input_ids"]
    column_token_indices = [] 
    for tokenized_column_ids in tokenized_columns_ids:
        # Find the index of row_ids +1'th [C_SEP] token
        try:
            column_token_indices.append(torch.where(tokenized_column_ids == csep_index)[0][row_idx])
        except IndexError:
            # In this case, the column is way too long (It is very rare)
            # Here, we just assign the index of [CLS] token
            column_token_indices.append(torch.tensor(0))
    column_token_indices = torch.stack(column_token_indices)
    return column_token_indices

def convert_tb_to_features_bert_metadata_threecat_with_rank(passages, table, meta_data, rank_info, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, text_pair=passage_str, max_length=args.max_q_len*2,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_metadata_norm_with_rank(table, passages, meta_data, rank_info, args.rank_scheme)
        else:
            table_str, passage_str = convert_tb_to_string_metadata_with_rank(table, passages, meta_data, rank_info, args.rank_scheme)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                       return_length=True)
    cls_id = tokenizer.cls_token_id 
    eos_id = tokenizer.eos_token_id 
    zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]


    inputs['part2_mask'] = torch.tensor([[0] +
                                         [1] * (zero_positions[1] - 1) +
                                         [0] * (inputs['length'] - zero_positions[1])])
    inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                         [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                         [0] * (inputs['length'] - zero_positions[3])])
    del inputs['special_tokens_mask']
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs



def convert_aug_tb_to_features_bert_threecat(passages, tokenizer, args):
    table_str, passage_str = passages.split('[PASSAGE]')
    passage_str = '[PASSAGE]' + passage_str
    inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                   add_special_tokens=True, padding='max_length',
                                   return_tensors='pt', truncation=True, return_special_tokens_mask=True,
                                   return_length=True)

    cls_id = tokenizer.cls_token_id 
    eos_id = tokenizer.eos_token_id 
    zero_positions = torch.where((inputs['input_ids'] == cls_id) | (inputs['input_ids'] == eos_id))[1]

    inputs['part2_mask'] = torch.tensor([[0] +
                                         [1] * (zero_positions[1] - 1) +
                                         [0] * (inputs['length'] - zero_positions[1])])
    inputs['part3_mask'] = torch.tensor([[0] * (zero_positions[2] + 1) +
                                         [1] * (zero_positions[3] - zero_positions[2] - 1) +
                                         [0] * (inputs['length'] - zero_positions[3])])

    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs


def convert_tb_to_features_bert_metadata_with_rank(passages, table, meta_data, tokenizer, args, encode=False):
    if table.empty:
        if type(passages) == str:
            passage_str = passages  # == query
        elif type(passages) == list:
            passage_str = ' '.join(passages)
        else:
            raise TypeError ("type passages: {}, {}, {}, {}".format(type(passages), passages, table, meta_data))
        inputs = tokenizer.encode_plus(passage_str, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        if args.normalize_table:
            # This is default option!
            # Here, normalize means we linearize the table as "{$header} is {$value}"
            table_str, passage_str = convert_tb_to_string_metadata_norm_with_rank(table, passages, meta_data)
        else:
            table_str, passage_str = convert_tb_to_string_metadata_with_rank(table, passages, meta_data)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    # inputs['length'] = len(inputs['input_ids'])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    # batch_outputs = BatchEncoding(inputs, tensor_type='pt', prepend_batch_axis=True)
    return batch_outputs




def convert_tb_to_features_bert(passages, table, tokenizer, args):
    if table.empty:
        inputs = tokenizer.encode_plus(passages, max_length=args.max_q_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    else:
        if args.normalize_table:
            table_str, passage_str = convert_tb_to_string_norm(table, passages)
        else:
            table_str, passage_str = convert_tb_to_string(table, passages)
        inputs = tokenizer.encode_plus(table_str, text_pair=passage_str, max_length=args.max_c_len,
                                       add_special_tokens=True, padding='max_length',
                                       return_tensors='pt', truncation=True, return_length=True)
    # inputs['length'] = len(inputs['input_ids'])
    batch_outputs = BatchEncoding(inputs, tensor_type='pt')
    return batch_outputs


def convert_tb_to_features_tapas(passage, table, tokenizer, args):
    # tokenizer: TapasTokenizer

    passage_tokens = tokenizer.tokenize(passage)
    tokenized_table = tokenizer._tokenize_table(table)

    num_rows = tokenizer._get_num_rows(table, drop_rows_to_fit=True)
    num_columns = tokenizer._get_num_columns(table)
    _, _, num_tokens = tokenizer._get_table_boundaries(tokenized_table)

    table_data = list(tokenizer._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))
    table_tokens = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))

    passage_ids = tokenizer.convert_tokens_to_ids(passage_tokens)
    table_ids = tokenizer.convert_tokens_to_ids(table_tokens)
    # input_ids = passage_ids + table_ids
    # input_ids = tokenizer.build_inputs_with_special_tokens(passage_ids, table_ids)
    # max_c_len = 512
    if len(passage_ids + table_ids) > args.max_c_len - 2:
        length_passage = max(args.max_c_len - len(table_ids) - 2, 100)
        length_table = min(410, len(table_ids))
        input_ids = [tokenizer.cls_token_id] + passage_ids[:length_passage] + \
                    [tokenizer.sep_token_id] + table_ids[:length_table]
        input_ids = input_ids[:args.max_c_len]
        passage_ids = passage_ids[:length_passage]
        passage = tokenizer.decode(passage_ids[:length_passage])
    else:
        input_ids = [tokenizer.cls_token_id] + passage_ids + [tokenizer.sep_token_id] + table_ids
        input_ids = input_ids[:args.max_c_len]

    encoded_inputs = {}
    encoded_inputs['input_ids'] = input_ids
    segment_ids = tokenizer.create_segment_token_type_ids_from_sequences(passage_ids, table_data)
    column_ids = tokenizer.create_column_token_type_ids_from_sequences(passage_ids, table_data)
    row_ids = tokenizer.create_row_token_type_ids_from_sequences(passage_ids, table_data)
    prev_labels = [0] * len(row_ids)

    raw_table = add_numeric_table_values(table)
    raw_passage = add_numeric_values_to_question(passage)
    column_ranks, inv_column_ranks = tokenizer._get_numeric_column_ranks(column_ids, row_ids, raw_table)
    numeric_relations = tokenizer._get_numeric_relations(raw_passage, column_ids, row_ids, raw_table)
    # if answer_coordinates is not None and answer_text is not None:
    #     labels = self.get_answer_ids(column_ids, row_ids, table_data, answer_text, answer_coordinates)
    #     numeric_values = self._get_numeric_values(raw_table, column_ids, row_ids)
    #     numeric_values_scale = self._get_numeric_values_scale(raw_table, column_ids, row_ids)
    #     encoded_inputs["labels"] = labels
    #     encoded_inputs["numeric_values"] = numeric_values
    #     encoded_inputs["numeric_values_scale"] = numeric_values_scale

    token_type_ids = [
        segment_ids,
        column_ids,
        row_ids,
        prev_labels,
        column_ranks,
        inv_column_ranks,
        numeric_relations,
    ]

    token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
    encoded_inputs["token_type_ids"] = token_type_ids

    attention_mask = tokenizer.create_attention_mask_from_sequences(passage_ids, table_data)
    encoded_inputs["attention_mask"] = attention_mask

    # Check lengths
    if len(encoded_inputs["input_ids"]) > tokenizer.model_max_length:
        logger.warning(
                f"Token indices sequence length is longer than the specified maximum sequence length "
                f"for this model ({len(encoded_inputs['input_ids'])} > {tokenizer.model_max_length}). Running this "
                "sequence through the model will result in indexing errors.")

    # Padding
    encoded_inputs = tokenizer.pad(encoded_inputs, max_length=args.max_c_len, padding='max_length',
                                   return_attention_mask=True)
    encoded_inputs["length"] = len(encoded_inputs["input_ids"])

    batch_outputs = BatchEncoding(encoded_inputs, tensor_type='pt', prepend_batch_axis=True)

    return batch_outputs


def convert_tb_to_features_tapas_metadata(passage, table, meta_data, tokenizer, args):
    # tokenizer: TapasTokenizer
    passage = ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + ' [PASSAGE] ' + passage

    passage_tokens = tokenizer.tokenize(passage)
    tokenized_table = tokenizer._tokenize_table(table)

    num_rows = tokenizer._get_num_rows(table, drop_rows_to_fit=True)
    num_columns = tokenizer._get_num_columns(table)
    _, _, num_tokens = tokenizer._get_table_boundaries(tokenized_table)

    table_data = list(tokenizer._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))
    table_tokens = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))

    passage_ids = tokenizer.convert_tokens_to_ids(passage_tokens)
    table_ids = tokenizer.convert_tokens_to_ids(table_tokens)
    # input_ids = passage_ids + table_ids
    # input_ids = tokenizer.build_inputs_with_special_tokens(passage_ids, table_ids)
    # max_c_len = 512
    if len(passage_ids + table_ids) > args.max_c_len - 2:
        length_passage = max(args.max_c_len - len(table_ids) - 2, 100)
        length_table = min(410, len(table_ids))
        input_ids = [tokenizer.cls_token_id] + passage_ids[:length_passage] + \
                    [tokenizer.sep_token_id] + table_ids[:length_table]
        input_ids = input_ids[:args.max_c_len]
        passage_ids = passage_ids[:length_passage]
        passage = tokenizer.decode(passage_ids[:length_passage])
    else:
        input_ids = [tokenizer.cls_token_id] + passage_ids + [tokenizer.sep_token_id] + table_ids
        input_ids = input_ids[:args.max_c_len]

    encoded_inputs = {}
    encoded_inputs['input_ids'] = input_ids
    segment_ids = tokenizer.create_segment_token_type_ids_from_sequences(passage_ids, table_data)
    column_ids = tokenizer.create_column_token_type_ids_from_sequences(passage_ids, table_data)
    row_ids = tokenizer.create_row_token_type_ids_from_sequences(passage_ids, table_data)
    prev_labels = [0] * len(row_ids)

    raw_table = add_numeric_table_values(table)
    raw_passage = add_numeric_values_to_question(passage)
    column_ranks, inv_column_ranks = tokenizer._get_numeric_column_ranks(column_ids, row_ids, raw_table)
    numeric_relations = tokenizer._get_numeric_relations(raw_passage, column_ids, row_ids, raw_table)

    token_type_ids = [
        segment_ids,
        column_ids,
        row_ids,
        prev_labels,
        column_ranks,
        inv_column_ranks,
        numeric_relations,
    ]

    token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
    encoded_inputs["token_type_ids"] = token_type_ids

    attention_mask = tokenizer.create_attention_mask_from_sequences(passage_ids, table_data)
    encoded_inputs["attention_mask"] = attention_mask

    # Check lengths
    if len(encoded_inputs["input_ids"]) > tokenizer.model_max_length:
        logger.warning(
                f"Token indices sequence length is longer than the specified maximum sequence length "
                f"for this model ({len(encoded_inputs['input_ids'])} > {tokenizer.model_max_length}). Running this "
                "sequence through the model will result in indexing errors.")

    # Padding
    encoded_inputs = tokenizer.pad(encoded_inputs, max_length=args.max_c_len, padding='max_length',
                                   return_attention_mask=True)
    encoded_inputs["length"] = len(encoded_inputs["input_ids"])

    batch_outputs = BatchEncoding(encoded_inputs, tensor_type='pt', prepend_batch_axis=True)

    return batch_outputs


class TRDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 args,
                 train=False,
                 ):
        super().__init__()
        # self.tokenizer = tokenizer
        # self.args = args
        self.train = train
        self.question = []
        self.table_block = []
        self.labels = []

        logger.info(f"Loading data from {data_path}")
        self.data = []
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        logger.info(f"Total sample count {len(self.data)}")

        for js in tqdm(self.data, desc='preparing dataset..'):
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]
            # self.table_block.append(convert_tb_to_features_tapas(' '.join(js['passages']), js['table'], tokenizer=tokenizer, args=args))
            # self.question.append(convert_tb_to_features_tapas(question, pd.DataFrame([]), tokenizer=tokenizer, args=args))
            # self.table_block.append(convert_tb_to_features(' '.join(js['passages']), js['table'], tokenizer=tokenizer, args=args))
            # self.question.append(convert_tb_to_features(question, pd.DataFrame([]), tokenizer=tokenizer, args=args))
            self.labels.append(torch.tensor(js['label']))

    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.data)


class TRDatasetNega(Dataset):

    def __init__(self, tokenizer, data_path, args, train=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.args = args
        self.train = train
        self.question = []
        self.table_block = []
        self.negative_table_block = []
        self.labels = []
        self.data = []

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        logger.info(f"Debugging mode: {len(self.data)}")
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            for js in tqdm(self.data, desc='preparing singleRetriever dataset..'):
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in self.args.model_name:
                    self.question.append(convert_tb_to_features_tapas(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args))
                    self.table_block.append(convert_tb_to_features_tapas(' '.join(js['passages']), js['table'], tokenizer=self.tokenizer, args=self.args))
                    self.negative_table_block.append(convert_tb_to_features_tapas(' '.join(js['neg_passages']), js['neg_table'], tokenizer=self.tokenizer, args=self.args))
                else:
                    self.question.append(convert_tb_to_features_bert(question, pd.DataFrame([]), tokenizer=self.tokenizer, args=self.args))
                    # self.question.append(tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True))
                    self.table_block.append(convert_tb_to_features_bert(js['passages'], js['table'],
                                                                        tokenizer=self.tokenizer, args=self.args))
                    self.negative_table_block.append(convert_tb_to_features_bert(js['neg_passages'], js['neg_table'],
                                                                                 tokenizer=self.tokenizer, args=self.args))

                self.labels.append(torch.tensor(js['label']))
            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "neg_tb": self.negative_table_block[index],
            "label": self.labels[index],
        }

    def __len__(self):
        return len(self.question)

    def save_tensor(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.question, os.path.join(path, 'question.pkl'))
        logger.info("Saving training question \t tensors to {}".format(os.path.join(path, 'question.pkl')))
        torch.save(self.table_block, os.path.join(path, 'table_block.pkl'))
        logger.info("Saving training table_block \t tensors to {}".format(os.path.join(path, 'table_block.pkl')))
        torch.save(self.negative_table_block, os.path.join(path, 'negative_table_block.pkl'))
        logger.info("Saving training negative_table_block \t tensors to {}".format(
            os.path.join(path, 'negative_table_block.pkl')))
        torch.save(self.labels, os.path.join(path, 'labels.pkl'))
        logger.info("Saving training labels \t tensors to {}".format(os.path.join(path, 'labels.pkl')))
        logger.info("Saving training (q, tb, neg_tb, lab) tensors to {}".format(path))

    def load_tensor(self, path):
        self.question = torch.load(os.path.join(path, 'question.pkl'))
        logger.info("Loading training question \t tensors from {}".format(os.path.join(path, 'question.pkl')))
        self.table_block = torch.load(os.path.join(path, 'table_block.pkl'))
        logger.info("Loading training table_block \t tensors from {}".format(os.path.join(path, 'table_block.pkl')))
        self.negative_table_block = torch.load(os.path.join(path, 'negative_table_block.pkl'))
        logger.info("Loading training negative_table_block \t tensors from {}".format(
            os.path.join(path, 'negative_table_block.pkl')))
        self.labels = torch.load(os.path.join(path, 'labels.pkl'))
        logger.info("Loading training labels \t tensors from {}".format(os.path.join(path, 'labels.pkl')))
        logger.info("Loading training (q, tb, neg_tb, lab) tensors from {}".format(path))


class TRDatasetNegaMeta(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaMeta, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    output['question'] = convert_tb_to_features_tapas_metadata(question, pd.DataFrame([]), js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = convert_tb_to_features_tapas_metadata(psg, js['table'], js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = convert_tb_to_features_tapas_metadata(psg, js['neg_table'], js['meta_data'], tokenizer=tokenizer, args=args)
                else:
                    output['question'] = convert_tb_to_features_bert_metadata(question, pd.DataFrame([]), js['meta_data'],
                                                                              tokenizer=tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = convert_tb_to_features_bert_metadata(psg, js['table'], js['meta_data'],
                                                                        tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = convert_tb_to_features_bert_metadata(psg, js['neg_table'], js['meta_data'],
                                                                                 tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output
            logger.info("normalize table text")
            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)


class TRDatasetNegaWithMetaRank(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaWithMetaRank, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        # self.data = self.data[:300]
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
            os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                """
                if 'tapas' in args.model_name:
                    output['question'] = convert_tb_to_features_tapas_metadata(question, pd.DataFrame([]), js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = convert_tb_to_features_tapas_metadata(psg, js['table'], js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = convert_tb_to_features_tapas_metadata(psg, js['neg_table'], js['meta_data'], tokenizer=tokenizer, args=args)
                """
                assert "tapas" not in args.model_name 
                output['question'] = convert_tb_to_features_bert_metadata_with_rank(question, pd.DataFrame([]), js['meta_data'],
                                                                            tokenizer=tokenizer, args=args)
                # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                #                                            return_tensors='pt', truncation=True, padding=True)
                psg = get_passages(js, args.psg_mode, neg=False)[:8]
                output['table_block'] = convert_tb_to_features_bert_metadata_with_rank(psg, js['table'], js['meta_data'],
                                                                    tokenizer=tokenizer, args=args)
                psg = get_passages(js, args.psg_mode, neg=True)[:8]
                output['negative_table_block'] = convert_tb_to_features_bert_metadata_with_rank(psg, js['neg_table'], js['meta_data'],
                                                                                 tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output
        
            logger.info("normalize table text")
            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever dataset with metadata and rank information..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)



class TRDatasetNegaMetaThreeCat(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaMetaThreeCat, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        print(os.path.abspath(self.data_path))
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        # self.data = self.data[:500]
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
            os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    raise Exception ('did not implement tapas with threecat setting')
                else:
                    if args.one_query:
                        function = convert_tb_to_features_bert_metadata_threecat_one_query
                    else:
                        function = convert_tb_to_features_bert_metadata_threecat
                    output['question'] = function(question, pd.DataFrame([]), js['meta_data'], tokenizer=tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = function(psg, js['table'], js['meta_data'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = function(psg, js['neg_table'], js['meta_data'], tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output

            logger.info("normalize table text")
            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever ThreeCat dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def add_augment_data(self, file):
        augment_data = load_jsonl(file)
        # augment_data = augment_data[:500]

        def running_function(js, args, tokenizer):
            output = {}
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]

            if args.one_query:
                function = convert_tb_to_features_bert_metadata_threecat_one_query
            else:
                function = convert_tb_to_features_bert_metadata_threecat
            output['question'] = function(question, pd.DataFrame([]), None, tokenizer=tokenizer, args=args)
            output['table_block'] = convert_aug_tb_to_features_bert_threecat(js['passages'], tokenizer=tokenizer, args=args)
            output['negative_table_block'] = convert_aug_tb_to_features_bert_threecat(js['neg_passages'], tokenizer=tokenizer, args=args)
            output['labels'] = torch.tensor(js['label'])
            return output
        for js in tqdm(augment_data, total=len(augment_data),
                       desc="preparing augment_data dataset threecat with metadata..", ):
            res = running_function(js, self.args, self.tokenizer)
            self.question.append(res['question'])
            self.table_block.append(res['table_block'])
            self.negative_table_block.append(res['negative_table_block'])
            self.labels.append(res['labels'])

        logger.info("after augmentation: num: {}".format(len(self.question)))

        # if self.train and self.args.save_tensor_path:
        #     self.save_tensor(self.args.save_tensor_path)


class TRDatasetNegaWithGlobalMetaThreecat(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaWithGlobalMetaThreecat, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        print(os.path.abspath(self.data_path))
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        logger.info(f"Total sample count {len(self.data)}")

        logger.info(f"Loading blink tables from {self.args.all_blink_table_path}")
        with open(self.args.all_blink_table_path, 'r') as f: 
            self.all_tables = json.load(f)
        # Summary token selection
        if "roberta" in self.args.model_name:
            summary_token = "madeupword0000"
        elif "bert" in self.args.model_name: 
            summary_token = "[unused0]"
        else:
            raise ValueError("Model not supported for summary injection")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if js["table_id"] in self.all_tables:
                    whole_table = self.all_tables[js["table_id"]]
                else:
                    whole_table = None 
                
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    raise Exception ('did not implement tapas with threecat setting')
                else:
                    if args.one_query:
                        # default options 
                        function = convert_tb_to_features_bert_metadata_threecat_with_global_one_query
                    else:
                        # not implemented 
                        raise NotImplementedError
                    output['question'] = function(question, pd.DataFrame([]), js['meta_data'], None, None, tokenizer=tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = function(psg, js['table'], js['meta_data'], summary_token, whole_table, tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = function(psg, js['neg_table'], js['meta_data'], summary_token, whole_table, tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output

            logger.info("normalize table text")
            for js in tqdm(self.data, total=len(self.data), desc="preparing Global ThreeCat dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def add_augment_data(self, file):
        raise NotImplementedError




class TRDatasetNegaWithGlobalColumnMetaThreecat(TRDatasetNega):

    def __init__(self, column_model_tokenizer, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaWithGlobalColumnMetaThreecat, self).__init__(tokenizer, data_path, args, train)
        self.column_model_tokenizer = column_model_tokenizer 

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        print(os.path.abspath(self.data_path))
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        
        logger.info(f"Total sample count: {len(self.data)}")

        logger.info(f"Loading blink tables from {self.args.all_blink_table_path}")
        with open(self.args.all_blink_table_path, 'r') as f: 
            self.all_tables = json.load(f)
        # column_summary_token token selection
        if "roberta" in self.args.model_name:
            column_summary_token = "madeupword0000"
        elif "bert" in self.args.model_name: 
            column_summary_token = "[unused0]"
        else:
            raise ValueError("Model not supported for summary injection")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path is not None and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer, column_model_tokenizer):
                output = {}
                question = js['question']
                if js["table_id"] in self.all_tables:
                    whole_table = self.all_tables[js["table_id"]]
                else:
                    whole_table = None 
                
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    raise Exception ('did not implement tapas with threecat setting')
                else:
                    if args.one_query:
                        # default options 
                        function = convert_tb_to_features_bert_metadata_threecat_with_column_global_one_query
                    else:
                        # not implemented 
                        raise NotImplementedError
                    output['question'] = function(question, pd.DataFrame([]), None, None, None, tokenizer=tokenizer, column_model_tokenizer=column_model_tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = function(psg, js['table'], js['meta_data'], column_summary_token, whole_table, tokenizer=tokenizer, column_model_tokenizer=column_model_tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = function(psg, js['neg_table'], js['meta_data'], column_summary_token, whole_table, tokenizer=tokenizer, column_model_tokenizer=column_model_tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output

            logger.info("normalize table text / with consideration of columm information ")
            for js in tqdm(self.data, total=len(self.data), desc="preparing Global ColumnMetaThreecat dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer, self.column_model_tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path is not None:
                logger.info("Saving!!!!")
                self.save_tensor(self.args.save_tensor_path)


    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "neg_tb": self.negative_table_block[index],
            "label": self.labels[index],
        }

    def add_augment_data(self, file):
        raise NotImplementedError




class TRDatasetNegaWithFusionColumnMetaThreecat(TRDatasetNega):

    def __init__(self, column_model_tokenizer, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaWithFusionColumnMetaThreecat, self).__init__(tokenizer, data_path, args, train)
        self.column_model_tokenizer = column_model_tokenizer 

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        print(os.path.abspath(self.data_path))
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        
        logger.info(f"Total sample count: {len(self.data)}")

        logger.info(f"Loading blink tables from {self.args.all_blink_table_path}")
        with open(self.args.all_blink_table_path, 'r') as f: 
            self.all_tables = json.load(f)

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path is not None and \
                os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer, column_model_tokenizer):
                output = {}
                question = js['question']
                if js["table_id"] in self.all_tables:
                    whole_table = self.all_tables[js["table_id"]]
                else:
                    whole_table = None 
                
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    raise Exception ('did not implement tapas with threecat setting')
                else:
                    if args.one_query:
                        function = convert_tb_to_features_bert_metadata_threecat_with_column_fusion_one_query
                    else:
                        # not implemented 
                        raise NotImplementedError
                    output['question'] = function(question, pd.DataFrame([]), None, None, tokenizer=tokenizer, column_model_tokenizer=column_model_tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = function(psg, js['table'], js['meta_data'], whole_table, tokenizer=tokenizer, column_model_tokenizer=column_model_tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = function(psg, js['neg_table'], js['meta_data'],  whole_table, tokenizer=tokenizer, column_model_tokenizer=column_model_tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output

            logger.info("normalize table text / with consideration of columm information ")
            for js in tqdm(self.data, total=len(self.data), desc="preparing Global ColumnMetaThreecat dataset with metadata..", ):
                res = running_function(js, self.args, self.tokenizer, self.column_model_tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path is not None:
                logger.info("Saving!!!!")
                self.save_tensor(self.args.save_tensor_path)


    def __getitem__(self, index):
        return {
            "q": self.question[index],
            "tb": self.table_block[index],
            "neg_tb": self.negative_table_block[index],
            "label": self.labels[index],
        }

    def add_augment_data(self, file):
        raise NotImplementedError




class TRDatasetNegaWithMetaRankThreeCat(TRDatasetNega):

    def __init__(self, tokenizer, data_path, args, train=False):
        super(TRDatasetNegaWithMetaRankThreeCat, self).__init__(tokenizer, data_path, args, train)

    def processing_data(self):
        logger.info(f"Loading data from {self.data_path}")
        print(os.path.abspath(self.data_path))
        with open(self.data_path, 'rb') as f:
            self.data = list(pickle.load(f))
        logger.info(f"Total sample count {len(self.data)}")

        if not self.args.overwrite_cache and self.train and self.args.save_tensor_path and \
            os.path.exists(self.args.save_tensor_path) and len(os.listdir(self.args.save_tensor_path)) == 4:
            self.load_tensor(self.args.save_tensor_path)
        else:
            def running_function(js, args, tokenizer):
                output = {}
                question = js['question']
                if question.endswith("?"):
                    question = question[:-1]
                if 'tapas' in args.model_name:
                    raise Exception ('did not implement tapas with threecat setting')
                else:
                    if args.one_query:
                        function = convert_tb_to_features_bert_metadata_threecat_with_rank_one_query
                    else:
                        function = convert_tb_to_features_bert_metadata_threecat_with_rank
                    output['question'] = function(question, pd.DataFrame([]), js['meta_data'], None, tokenizer=tokenizer, args=args)
                    # output['question'] = tokenizer.encode_plus(question, max_length=args.max_q_len,
                    #                                            return_tensors='pt', truncation=True, padding=True)
                    psg = get_passages(js, args.psg_mode, neg=False)[:8]
                    output['table_block'] = function(psg, js['table'], js['meta_data'], js['pos_sample_rank_list'], tokenizer=tokenizer, args=args)
                    psg = get_passages(js, args.psg_mode, neg=True)[:8]
                    output['negative_table_block'] = function(psg, js['neg_table'], js['meta_data'], js['neg_sample_rank_list'], tokenizer=tokenizer, args=args)
                output['labels'] = torch.tensor(js['label'])
                return output

            logger.info("normalize table text")

            for js in tqdm(self.data, total=len(self.data), desc="preparing singleRetriever ThreeCat dataset with metadata and rank information..", ):
                res = running_function(js, self.args, self.tokenizer)
                self.question.append(res['question'])
                self.table_block.append(res['table_block'])
                self.negative_table_block.append(res['negative_table_block'])
                self.labels.append(res['labels'])

            if self.train and self.args.save_tensor_path:
                self.save_tensor(self.args.save_tensor_path)

    def add_augment_data(self, file):
        augment_data = load_jsonl(file)
        # augment_data = augment_data[:500]

        def running_function(js, args, tokenizer):
            output = {}
            question = js['question']
            if question.endswith("?"):
                question = question[:-1]

            if args.one_query:
                function = convert_tb_to_features_bert_metadata_threecat_one_query
            else:
                function = convert_tb_to_features_bert_metadata_threecat
            output['question'] = function(question, pd.DataFrame([]), None, tokenizer=tokenizer, args=args)
            output['table_block'] = convert_aug_tb_to_features_bert_threecat(js['passages'], tokenizer=tokenizer, args=args)
            output['negative_table_block'] = convert_aug_tb_to_features_bert_threecat(js['neg_passages'], tokenizer=tokenizer, args=args)
            output['labels'] = torch.tensor(js['label'])
            return output

        for js in tqdm(augment_data, total=len(augment_data),
                       desc="preparing augment_data dataset threecat with metadata..", ):
            res = running_function(js, self.args, self.tokenizer)
            self.question.append(res['question'])
            self.table_block.append(res['table_block'])
            self.negative_table_block.append(res['negative_table_block'])
            self.labels.append(res['labels'])

        logger.info("after augmentation: num: {}".format(len(self.question)))

        # if self.train and self.args.save_tensor_path:
        #     self.save_tensor(self.args.save_tensor_path)




def collate_all_tokens(input_type, inputs, *args):
    return_dict = {}
    for arg in args:
        return_dict[input_type + '_' + arg] = collate_tokens([input[input_type][arg].view(-1) for input in inputs], 0)
    return return_dict



def tb_collate_tapas(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'q_input_ids': collate_tokens([s["q"]["input_ids"].view(-1) for s in samples], pad_id),
        'q_mask': collate_tokens([s["q"]["attention_mask"].view(-1) for s in samples], 0),
        'c_input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'c_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
        'neg_input_ids': collate_tokens([s["neg_tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'neg_mask': collate_tokens([s["neg_tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["q"]:
        batch.update({
            'q_part2_mask': collate_tokens([s["q"]["part2_mask"].view(-1) for s in samples], 0),
            'c_part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
            'neg_part2_mask': collate_tokens([s["neg_tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["q"]:
        batch.update({
            'q_part3_mask': collate_tokens([s["q"]["part3_mask"].view(-1) for s in samples], 0),
            'c_part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
            'neg_part3_mask': collate_tokens([s["neg_tb"]["part3_mask"].view(-1) for s in samples], 0),
        })

    if "token_type_ids" in samples[0]["q"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q"]["token_type_ids"] for s in samples], 0,'2d'),
            'c_type_ids': collate_tokens([s["tb"]["token_type_ids"] for s in samples], 0,'2d'),
            'neg_type_ids': collate_tokens([s["neg_tb"]["token_type_ids"] for s in samples], 0,'2d'),
        })

    return batch


def tb_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'q_input_ids': collate_tokens([s["q"]["input_ids"].view(-1) for s in samples], pad_id),
        'q_mask': collate_tokens([s["q"]["attention_mask"].view(-1) for s in samples], 0),
        'c_input_ids': collate_tokens([s["tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'c_mask': collate_tokens([s["tb"]["attention_mask"].view(-1) for s in samples], 0),
        'neg_input_ids': collate_tokens([s["neg_tb"]["input_ids"].view(-1) for s in samples], pad_id),
        'neg_mask': collate_tokens([s["neg_tb"]["attention_mask"].view(-1) for s in samples], 0),
    }

    if "part2_mask" in samples[0]["q"]:
        batch.update({
            'q_part2_mask': collate_tokens([s["q"]["part2_mask"].view(-1) for s in samples], 0),
            'c_part2_mask': collate_tokens([s["tb"]["part2_mask"].view(-1) for s in samples], 0),
            'neg_part2_mask': collate_tokens([s["neg_tb"]["part2_mask"].view(-1) for s in samples], 0),
        })

    if "part3_mask" in samples[0]["q"]:
        batch.update({
            'q_part3_mask': collate_tokens([s["q"]["part3_mask"].view(-1) for s in samples], 0),
            'c_part3_mask': collate_tokens([s["tb"]["part3_mask"].view(-1) for s in samples], 0),
            'neg_part3_mask': collate_tokens([s["neg_tb"]["part3_mask"].view(-1) for s in samples], 0),
        })
    
    if "table_input_ids" in samples[0]["q"]:
        batch.update({
            "c_table_input_ids": collate_tokens([s["tb"]["table_input_ids"].view(-1) for s in samples], pad_id),
            "c_table_mask": collate_tokens([s["tb"]["table_mask"].view(-1) for s in samples], 0),
            "neg_table_input_ids": collate_tokens([s["neg_tb"]["table_input_ids"].view(-1) for s in samples], pad_id),
            "neg_table_mask": collate_tokens([s["neg_tb"]["table_mask"].view(-1) for s in samples], 0),

        })


    if "token_type_ids" in samples[0]["q"]:
        batch.update({
            'q_type_ids': collate_tokens([s["q"]["token_type_ids"] for s in samples], 0, '2d'),
            'c_type_ids': collate_tokens([s["tb"]["token_type_ids"] for s in samples], 0, '2d'),
            'neg_type_ids': collate_tokens([s["neg_tb"]["token_type_ids"] for s in samples], 0, '2d'),
            "c_table_type_ids": collate_tokens([s["tb"]["table_token_type_ids"] for s in samples], 0, '2d'),
            "neg_table_type_ids": collate_tokens([s["neg_tb"]["table_token_type_ids"] for s in samples], 0, '2d'),
        })
    
    if "column_input_ids_list" in samples[0]["tb"]:
        batch.update({
            'c_column_input_ids_list': collate_tokens_to_3d([s["tb"]["column_input_ids_list"] for s in samples], -1),
            'c_column_mask_list': collate_tokens_to_3d([s["tb"]["column_attention_mask_list"] for s in samples], -1),
            'c_column_token_indices_list': collate_tokens([s["tb"]["column_token_indices"].view(-1) for s in samples], -1), # This is for Column-level summary generation model 
            'neg_column_input_ids_list': collate_tokens_to_3d([s["neg_tb"]["column_input_ids_list"] for s in samples], -1),
            'neg_column_mask_list': collate_tokens_to_3d([s["neg_tb"]["column_attention_mask_list"] for s in samples], -1), 
            'neg_column_token_indices_list': collate_tokens([s["neg_tb"]["column_token_indices"].view(-1) for s in samples], -1), # This is for Column-level summary generation model 
        })
    
    if "value_mask" in samples[0]["tb"]:
        batch.update({
            "c_value_mask": collate_tokens([s["tb"]["value_mask"].view(-1) for s in samples], 0),
            "neg_value_mask": collate_tokens([s["neg_tb"]["value_mask"].view(-1) for s in samples], 0),
        })
    
    if "column_categories" in samples[0]["tb"]:
        batch.update({
            'c_column_categories': collate_tokens([s["tb"]["column_categories"].view(-1) for s in samples], -1),
            'neg_column_categories': collate_tokens([s["neg_tb"]["column_categories"].view(-1) for s in samples], -1),
        })
    return batch



