
import json
import pandas as pd
import torch 
import logging 
from tqdm import tqdm
from torch.utils.data import Dataset
from preprocessing.utils_preprocess import extract_value_and_type
from RATE.utils import getrank_column_direction

from transformers import BertForQuestionAnswering 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CSEP_ID = 50265

def construct_clsg_input_roberta(header, aggregated_column, title, section_title):
    """
    Input:
        header: a corresponding header for the column
        aggregated_column: a list of values in the column
        title: table title
        section_title: table section title
    Return:
        input_str: a string that is input to the column-level summary generation model
    """
    input_str = ' [TAB] ' + ' [TITLE] ' + title+' [SECTITLE] ' + section_title+ ' [DATA] [C_SEP] ' + \
                ' [C_SEP] '.join(['{} is {}.'.format(header, v) for v in aggregated_column])
    return input_str


class CLSummaryGenDataset(Dataset):
    """
    Column-level summary generation dataset 
    """

    def __init__(self, tokenizer, data_path, args):
        super().__init__()

        self.tokenizer = tokenizer
        self.data_path = data_path
        self.args = args
        self.objective = args.objective 
        logger.info(f"Objective: {args.objective}")
        assert len(self.tokenizer) == 50266

        logger.info("Loading wikitables data from {}".format(self.data_path))
        with open(self.data_path, "r") as f:
            self.wikitables = json.load(f)
        
        wikitable_ordered = {}
        keys = list(self.wikitables.keys())
        keys.sort() # We sort the keys to make the sampling deterministic

        for k in keys:
            wikitable_ordered[k] = self.wikitables[k]
        self.wikitables = wikitable_ordered
        self.tokenized_datapoints = []
        
        for k,v in tqdm(self.wikitables.items(), desc="Preprocessing data"):
            table = v 
            title = table["title"]
            section_title = table["section_title"]
            headers = table["header"]
            data = table["data"]

            aggregated_columns = [[] for i in range(len(headers))]

            for r in data:
                for j, value in enumerate(r):
                    aggregated_columns[j].append(value)
            
            for header, aggregated_column in zip(headers, aggregated_columns):
                if header =="":
                    # We ignore the column with empty header
                    continue 
                extracted_values, vtypes, ctype = extract_value_and_type(aggregated_column)
                assert len(extracted_values) == len(aggregated_column)
                if ctype == "numeric" or ctype == "date":
                    # We need to know the rank of the values in the column
                    cv_idx_ranks = getrank_column_direction(extracted_values, vtypes, ctype)
                    if cv_idx_ranks == -1:
                        # We ignore the column with invalid date values
                        continue

                    input_str = construct_clsg_input_roberta(header, aggregated_column, title, section_title)
                    tokenized = self.tokenizer(input_str, return_tensors="pt", padding="max_length", truncation=True, max_length=self.args.max_length)
                    if args.objective == "minmax":
                        input_ids = tokenized["input_ids"].squeeze()
                        csep_indices = torch.where(input_ids == CSEP_ID)[0]
                        max_value_rowidx = cv_idx_ranks[0][0]
                        min_value_rowidx = cv_idx_ranks[-1][0]
                        try:
                            max_value_csep_idx = csep_indices[max_value_rowidx]
                            min_value_csep_idx = csep_indices[min_value_rowidx]
                        except IndexError:
                            # It is possible that max or min value exists in truncated region
                            # In this case, we just ignore this datapoint.
                            continue
                        # First check if two indices are within the max_length
                        if (max_value_csep_idx < self.args.max_length) and (min_value_csep_idx < self.args.max_length):
                            # Then we can use this datapoint
                            # Squeeze the tokenized datapoints 
                            attention_mask = tokenized["attention_mask"].squeeze()
                            max_csep_index = torch.tensor([max_value_csep_idx])
                            min_csep_index = torch.tensor([min_value_csep_idx])
                            self.tokenized_datapoints.append(
                                {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                    "max_csep_index": max_csep_index,
                                    "min_csep_index": min_csep_index,
                                }
                            )
                    elif args.objective == "minmaxmedian":
                        input_ids = tokenized["input_ids"].squeeze()
                        csep_indices = torch.where(input_ids == CSEP_ID)[0]

                        num_values = len(cv_idx_ranks)
                        max_value_rowidx = cv_idx_ranks[0][0]
                        median_value_rowidx = cv_idx_ranks[num_values//2][0]
                        min_value_rowidx = cv_idx_ranks[-1][0]
                        try:
                            max_value_csep_idx = csep_indices[max_value_rowidx]
                            min_value_csep_idx = csep_indices[min_value_rowidx]
                            median_value_csep_idx = csep_indices[median_value_rowidx]
                        except IndexError:
                            # It is possible that max or min value exists in truncated region
                            # In this case, we just ignore this datapoint.
                            continue
                        # First check if two indices are within the max_length
                        if (max_value_csep_idx < self.args.max_length) and (min_value_csep_idx < self.args.max_length) and (median_value_csep_idx < self.args.max_length):
                            # Then we can use this datapoint
                            # Squeeze the tokenized datapoints 
                            attention_mask = tokenized["attention_mask"].squeeze()
                            max_csep_index = torch.tensor([max_value_csep_idx])
                            min_csep_index = torch.tensor([min_value_csep_idx])
                            median_csep_index = torch.tensor([median_value_csep_idx])
                            self.tokenized_datapoints.append(
                                {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                    "max_csep_index": max_csep_index,
                                    "min_csep_index": min_csep_index,
                                    "median_csep_index": median_csep_index,
                                }
                            )
                    elif args.objective =="median":
                        input_ids = tokenized["input_ids"].squeeze()
                        csep_indices = torch.where(input_ids == CSEP_ID)[0]

                        num_values = len(cv_idx_ranks)
                        median_value_rowidx = cv_idx_ranks[num_values//2][0]
                        try:
                            median_value_csep_idx = csep_indices[median_value_rowidx]
                        except IndexError:
                            # It is possible that max or min value exists in truncated region
                            # In this case, we just ignore this datapoint.
                            continue
                        # First check if two indices are within the max_length
                        if (median_value_csep_idx < self.args.max_length):
                            # Then we can use this datapoint
                            # Squeeze the tokenized datapoints 
                            attention_mask = tokenized["attention_mask"].squeeze()
                            median_csep_index = torch.tensor([median_value_csep_idx])
                            self.tokenized_datapoints.append(
                                {
                                    "input_ids": input_ids,
                                    "attention_mask": attention_mask,
                                    "median_csep_index": median_csep_index,
                                }
                            )

        logger.info(f"Total number of datapoints: {len(self.tokenized_datapoints)}")

        
    def __len__(self):
        return len(self.tokenized_datapoints)
    
    def __getitem__(self, index):
        return self.tokenized_datapoints[index]
        
                    


