import datetime
import pickle
import json
from tqdm import tqdm
import torch
from RATE.data.sg_dataset import construct_clsg_input_roberta

class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self.batch.to(device=self.device, non_blocking=True)


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch




def get_current_time_str():
    return str(datetime.now().strftime('%Y_%m_%d_%H:%M:%S'))
def load_json(filename):
    with open(filename,'r',encoding='utf8') as f:
        return json.load(f)

def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)
    return d_list

def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def save_json(results,filename):
    with open(filename,'w',encoding='utf8') as inf:
        json.dump(results,inf)

def convert_tb_to_string_metadata(table, passages, meta_data, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = '[TAB] ' + ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + ' [DATA] ' \
                + ' ; '.join(['{} is {}'.format(h, c) for h, c in zip(header, value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return '{} {}'.format(table_str, passage_str)

def convert_tb_to_string(table, passages, cut='passage', max_length=460, topk_block=15):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = '[HEADER] ' + ' '.join(['{} is {}'.format(h, c) for h, c in zip(header, value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    if cut == 'passage':
        table_length = min(max_length, len(table_str.split(' ')))
        doc_length = 0 if table_length >= max_length else max_length - table_length
    else:
        doc_length = min(max_length, len(passage_str.split(' ')))
        table_length = 0 if doc_length >= max_length else max_length - doc_length

    return '{} {}'.format(table_str, passage_str)

def convert_table_to_string(table, meta_data=None, max_length=90):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = '[HEADER] ' + ' '.join(['{} is {}'.format(h, c) for h, c in zip(header, value[0])])
    if meta_data:
        table_str = '[TAB] [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + ' ' + table_str
    return table_str

def convert_tb_to_string_norm(table, passages, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB] ' + ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header, value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str


def convert_tb_to_string_metadata_norm(table, passages, meta_data, cut='passage', max_length=400):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] ' + \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str


def convert_tb_to_string_metadata_with_summary_token(table, passages, meta_data, summary_token, cut='passage', max_length=400):
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB]' + summary_token + ' [TITLE] ' + meta_data['title'] + ' [SECTITLE] ' + meta_data['section_title'] + \
                ' [HEADER] ' + ' [SEP] '.join(header) + ' [DATA] '+' [SEP] '.join(value[0])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str

def convert_tb_to_string_metadata_norm_with_summary_token(table, passages, meta_data, summary_token, cut='passage', max_length=400):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    table_str = ' [TAB] ' +summary_token + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] ' + \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str

def convert_tb_to_string_metadata_norm_with_column_level_summary_token(table, passages, meta_data, summary_token, cut='passage', max_length=400):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    column_summary_strs = f' {summary_token} ' * len(header)
    table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ column_summary_strs + ' [DATA] ' + \
                ' '.join(['{} is {}.'.format(h,c) for h,c in zip(header,value[0])])
    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str




def convert_tb_to_string_metadata_with_rank(table, passages, meta_data, rank_info, rank_scheme, cut='passage', max_length=400):
    raise NotImplementedError("convert_tb_to_string_metadata_with_rank is not implemented yet")

def convert_tb_to_string_metadata_norm_with_rank(table, passages, meta_data, rank_info, rank_scheme, cut='passage', max_length=400):
    # normed text by processing table with " H1 is C1 .... "
    header = table.columns.tolist()
    value = table.values.tolist()
    if rank_info is None:
        table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] ' + \
                    ' '.join(['{} is {}. '.format(h,c) for h,c in zip(header,value[0])])
    else:
        table_str = ' [TAB] ' + ' [TITLE] ' + meta_data['title']+' [SECTITLE] ' + meta_data['section_title']+ ' [DATA] '
        rank_infused_data_str = ""

        if rank_scheme == "ctx_minmax":
            # Just like what "OTT-QA" paper did.
            for h,c,r in zip(header,value[0],rank_info):
                if r is not None:
                    rank = r["rank"]
                    out_of = r["out_of"]
                    if rank == 1:
                        # Min rank (we sort with ascending order.)
                        rank_infused_data_str += '[MIN] {} is {}. '.format(h,c)
                    elif rank == out_of:
                        # Max rank (we sort with ascending order.)
                        rank_infused_data_str += '[MAX] {} is {}. '.format(h,c)
                    else:
                        rank_infused_data_str += '{} is {}. '.format(h,c)
                else:
                    rank_infused_data_str += '{} is {}. '.format(h,c)
        elif rank_scheme == "ctx_singlerank":
            # We just add single rank information.
            for h,c,r in zip(header, value[0], rank_info):
                if r is not None:
                    rank = r["rank"]
                    rank_infused_data_str += '{} is {} with rank {}. '.format(h,c,rank)
                else:
                    rank_infused_data_str += '{} is {}. '.format(h,c)

        elif rank_scheme == "ctx_pairrank":
            # Pairwise rank information.
            # X is Y with rank 3 out of 10th. 
            for h,c,r in zip(header, value[0], rank_info):
                if r is not None:
                    rank = r["rank"]
                    out_of = r["out_of"]
                    rank_infused_data_str += '{} is {} with rank {} out of {}. '.format(h,c,rank, out_of)
                else:
                    rank_infused_data_str += '{} is {}. '.format(h,c)

        else:
            raise ValueError("rank_scheme: {} is not supported".format(rank_scheme))
        table_str += rank_infused_data_str

    passage_str = ' [PASSAGE] ' + ' [SEP] '.join(passages)
    return table_str, passage_str




def convert_whole_tb_to_string_norm(whole_table):
    # linearize whole table to string, in normalized format.
    # [HEADER] is [VALUE]
    title = whole_table["title"]
    section_title = whole_table["section_title"]
    header = [h[0] for h in whole_table["header"]]  
    rows = whole_table["data"]
    whole_table_str = ' [TAB] ' + ' [TITLE] ' + title+' [SECTITLE] ' + section_title+ ' [DATA] '

    data_str = ''
    for row in rows:
        row_values = [x[0] for x in row]
        for h, v in zip(header, row_values):
            data_str += '{} is {}. '.format(h,v)
        data_str += '\n'
    whole_table_str += data_str
    return whole_table_str


def convert_whole_tb_to_normalized_columns(whole_table):
    # linearize the whole table column-by-column 
    # This is for the column-level summary generation.
    title = whole_table["title"]
    section_title = whole_table["section_title"]
    headers = [h[0] for h in whole_table["header"]]  
    rows = whole_table["data"]

    aggregated_columns = [[] for i in range(len(headers))]
    for r in rows:
        for j, value in enumerate(r):
            cell_value = value[0]
            aggregated_columns[j].append(cell_value)

    column_level_inputs = []
    for header, aggregated_column in zip(headers, aggregated_columns):
        column_level_input = construct_clsg_input_roberta(header, aggregated_column, title, section_title)
        column_level_inputs.append(column_level_input)

    return column_level_inputs


def convert_whole_tb_to_string(whole_table):
    # linearize whole table to string 
    title = whole_table["title"]
    section_title = whole_table["section_title"]
    header = [h[0] for h in whole_table["header"]]  
    rows = whole_table["data"]
    whole_table_str = ' [TAB] ' + ' [TITLE] ' + title+' [SECTITLE] ' + section_title+ ' [HEADER] '

    data_str = ""
    # Append header information 
    for hidx, h in enumerate(header):
        if hidx != len(header)-1:
            data_str += f"{h}, "
        else:
            data_str += f"{h} [DATA] "
    
    for ridx, row in enumerate(rows):
        row_values = [x[0] for x in row]
        for v in row_values:
            data_str += '{} is {}. '.format(h,v)
        if ridx!=len(rows)-1:
            data_str += '\n'
    whole_table_str += data_str
    return whole_table_str



def get_passages(js, psg_mode, neg=False):
    # default psg_mode: ori
    prefix = "neg_" if neg else ""
    if psg_mode=='ori':
        psg = js[prefix+"passages"]
    elif psg_mode=="s_sent":
        psg = js[prefix+'s_sent'] if len(js[prefix+'s_sent']) > 0 else js[prefix+"passages"]
    elif psg_mode=="s_psg":
        psg = js[prefix+'s_psg'] if len(js[prefix+'s_psg']) > 0 else js[prefix+"passages"]
    else:
        psg = []
    return psg

