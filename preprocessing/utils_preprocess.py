from nltk.tokenize import word_tokenize, sent_tokenize
import urllib.parse
import sys
import json
import gzip
from dateutil.parser import parse
import re
import warnings
import numpy as np
from scipy.linalg import norm
from collections import Counter

#python qa_preprocess.py --split dev --retrieval_results_file ../../../data/preprocessed_data/qa/data4qa_v2/dev_output_k100.json  --reprocess
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from dateutil.parser import parse
import time 



def dict_list_sort_by_key(list,key,reverse=True):
    list.sort(key=lambda k: (k.get(key, 0)), reverse=reverse)
    return list
import copy
def rank_doc_tfidf(question, corpus):
    if corpus:
        if isinstance(corpus[0],dict):
            text_corpus = [question] + [doc['passage'] for doc in corpus]
        elif isinstance(corpus[0],str):
            text_corpus = [question] + copy.deepcopy(corpus)
            corpus = [{'passage': doc} for doc in corpus]
    else:
        text_corpus = [question]
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    vecs = cv.fit_transform(text_corpus).toarray()
    que_vec = vecs[0]

    for idx, doc_vec in enumerate(vecs[1:]):
        score = np.dot(que_vec, doc_vec) / (norm(que_vec) * norm(doc_vec))
        corpus[idx]['tf_score'] = score
    results = dict_list_sort_by_key(corpus, 'tf_score', reverse=True)
    return results
# def rank_doc_tfidf(question, corpus):
#
#     # if isinstance(corpus[0],dict):
#     #     text_corpus = [question] + [doc['passage'] for doc in corpus]
#     # elif isinstance(corpus[0],str):
#     #     text_corpus = copy.deepcopy(corpus)
#     #     corpus = [{'passage': doc} for doc in corpus]
#     text_corpus = [question] + [doc['passage'] for doc in corpus]
#     cv = TfidfVectorizer(tokenizer=lambda s: s.split())
#     vecs = cv.fit_transform(text_corpus).toarray()
#     que_vec = vecs[0]
#
#     for idx, doc_vec in enumerate(vecs[1:]):
#         score = np.dot(que_vec, doc_vec) / (norm(que_vec) * norm(doc_vec))
#         corpus[idx]['tf_score'] = score
#     results = dict_list_sort_by_key(corpus, 'tf_score', reverse=True)
#     return results


def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def write_jsonl(data, path):
    # data是一个包含多个dict的list
    with open(path, 'w') as fp:
        for inst in data:
            fp.write(json.dumps(inst) + '\n')
def read_jsonl(path):
    # data是一个包含多个dict的list
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            line = line.strip()
            data.append(json.loads(line))
    return data


def tokenize(string, remove_dot=False):
    def func(string):
        return " ".join(word_tokenize(string))

    string = string.replace('%-', '-')
    if remove_dot:
        string = string.rstrip('.')

    string = func(string)
    string = string.replace(' %', '%')
    return string


def url2dockey(string):
    string = urllib.parse.unquote(string)
    return string


def filter_firstKsents(string, k):
    string = sent_tokenize(string)
    string = string[:k]
    return " ".join(string)


def compressGZip(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    json_str = json.dumps(data) + "\n"  # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')  # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(file_name + '.gz', 'w') as fout:  # 4. gzip
        fout.write(json_bytes)


def readGZip(file_name):
    if file_name.endswith('gz'):
        with gzip.GzipFile(file_name, 'r') as fin:  # 4. gzip
            json_bytes = fin.read()  # 3. bytes (i.e. UTF-8)

        json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
        data = json.loads(json_str)  # 1. data
        return data
    else:
        with open(file_name, 'r') as fin:
            data = json.load(fin)
        return data

def extract_date(string):
    try:
        date = parse(string)
        return date
    except (ValueError, OverflowError):
        return None


def extract_num(string):
    # remove "," and " " in the string
    # if this is converted to float, then it is a number
    string = string.replace(',', '')
    string = string.replace(' ', '')
            
    try:
        num = float(string)
        return num
    except ValueError:
        return None

def extract_value_and_type(values):
    """
    Input:
        values: a list of values
    Return:
        ctype: whether the values are numeric, date, or string
    """
    # values: ['Lalith Athulathmudali', '1955', '1960', 'BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 )', 'President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993']
    # ctype: ['string', 'numeric', 'numeric', 'string', 'string']
    vtypes = []
    extracted_values = []
    for value in values:
        nv = extract_num(value)
        if nv is not None:
            extracted_values.append(nv)
            vtypes.append('numeric')
            continue 
        # Check if it is a range value 
        if '-' in value:
            value_splitted = value.split('-')[0]
            nv = extract_num(value_splitted)
            if nv is not None:
                extracted_values.append(nv)
                vtypes.append('range')
                continue

        dv = extract_date(value)
        if dv is not None:
            extracted_values.append(dv)
            vtypes.append('date')
            continue
        extracted_values.append(value)
        vtypes.append('string')    
    
    assert len(values) == len(extracted_values)

    vtypes_count = Counter(vtypes)
    if vtypes_count['numeric'] > vtypes_count['string'] and vtypes_count['numeric'] > vtypes_count['date']:
        ctype = 'numeric'
    elif vtypes_count['date'] > vtypes_count['string'] and vtypes_count['date'] > vtypes_count['numeric']:
        ctype = 'date'
    elif vtypes_count['string'] > vtypes_count['numeric'] and vtypes_count['string'] > vtypes_count['date']:
        ctype = 'string'
    else:
        ctype = 'range'
    
    return (extracted_values, vtypes, ctype)


def getrank_rowfield(row_idx, contents, header):
    """
    Input:
        row_idx: a idx of a row in a table
        contents: a list of rows 
        header: a list of headers
    Return:
        rank_list: a list of ranks of fields. If the field does not have rank, return "None". Otherwise, return dict with the following format:
        {
            'rank': 1,
            'out_of': 5,
        }
        Note that rank is 1-indexed.
    """
    # row: ['Lalith Athulathmudali', '1955', '1960', 'BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 )', 'President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993']
    # header: ['Name', 'M', 'G', 'Degree', 'Notes']


    # Aggregate the values corresponding to the same column
    rank_list = []
    aggregated_columns = [[] for i in range(len(header))]

    for r in contents:
        for j, value in enumerate(r):
            aggregated_columns[j].append(value)
    
    for aggregated_column in aggregated_columns:
        extracted_values, vtypes, ctype = extract_value_and_type(aggregated_column)

        current_field_value, current_field_type = extracted_values[row_idx], vtypes[row_idx]

        if current_field_type == "string":
            rank_list.append(None)
        else:
            if ctype == "numeric" and (current_field_type == "numeric" or current_field_type == "range"):
                # Get the rank of the current field
                # We filter values that are not numeric or range values
                filtered_values = [value for value, vtype in zip(extracted_values, vtypes) if vtype == "numeric" or vtype == "range"]
                filtered_values.sort()
                rank = filtered_values.index(current_field_value) + 1
                rank_list.append({"rank": rank, "out_of": len(filtered_values)})
            
            elif ctype == "date" and current_field_type == "date":
                filtered_values = [value for value, vtype in zip(extracted_values, vtypes) if vtype == "date"]
                try:
                    filtered_values.sort()
                    rank = filtered_values.index(current_field_value) + 1
                    rank_list.append({"rank": rank, "out_of": len(filtered_values)}) 
                except TypeError:
                    # TypeError occurs when the date has different time zone.
                    # In this case, we just don't assign the ranks.
                    # This happens very rarely.
                    rank_list.append(None)
                except ValueError:
                    # ValueError occurs 
                    # ValueError: offset must be a timedelta strictly between -timedelta(hours=24) and timedelta(hours=24).
                    # This happens very rarely and in this case we just don't assign the ranks.
                    rank_list.append(None)
            else:
                rank_list.append(None)

    return rank_list 


class CellHelper(object):
    """Cell Helper to detect the cell type."""

    @staticmethod
    def is_unit(string):
        """Is the input a unit."""
        return re.search(r'\b(kg|m|cm|lb|hz|million)\b', string.lower())

    @staticmethod
    def is_score(string):
        """Is the input a score between two things."""
        if re.search(r'[0-9]+ - [0-9]+', string):
            return True
        elif re.search(r'[0-9]+-[0-9]+', string):
            return True
        else:
            return False

    @staticmethod
    def is_date(string, fuzzy=False):
        """Is the input a date."""
        try:
            parse(string, fuzzy=fuzzy)
            return True
        except Exception:  # pylint: disable=broad-except
            return False

    @staticmethod
    def is_bool(string):
        if string.lower() in ['yes', 'no']:
            return True
        else:
            return False

    @staticmethod
    def is_float(string):
        if '.' in string:
            try:
                float(string)
                return True
            except Exception:
                return False
        else:
            return False

    @staticmethod
    def is_normal_word(string):
        if ' ' not in string:
            return string.islower()
        else:
            return False


def whitelist(string):
    """Is the input a whitelist string."""
    string = string.strip()
    if len(string) < 2:
        return False
    elif string.isdigit():
        if len(string) == 4:
            return True
        else:
            return False
    elif string.replace(',', '').isdigit():
        return False
    elif CellHelper.is_float(string):
        return False
    elif '#' in string or '%' in string or '+' in string or '$' in string:
        return False
    elif CellHelper.is_normal_word(string):
        return False
    elif CellHelper.is_bool(string):
        return False
    elif CellHelper.is_score(string):
        return False
    elif CellHelper.is_unit(string):
        return False
    elif CellHelper.is_date(string):
        return False
    return True


def is_year(string):
    if len(string) == 4 and string.isdigit():
        return True
    else:
        return False


def is_squence_number(string_list):
    string_list = list(filter(None, string_list))
    if all([string.isdigit() for string in string_list]):
        if all([len(string) != 4 for string in string_list]):
            l = [float(string) for string in string_list]
            if all(x <= y for x, y in zip(l, l[1:])):
                return True
    return False


def remove_null_header_column(header, contents):
    # 11993/419183个table中，存在header有空的情况，这11993个table中，空且为序号的有6720列, 全部去掉
    # 所以把为序号的列删掉，同时把非序号的列的空header改名字
    null_ids = [i for i, h in enumerate(header) if h == '']
    for i in reversed(null_ids):
        ret = [cont[i] for cont in contents]
        if is_squence_number(ret):
            [cont.pop(i) for cont in contents]
            header.pop(i)
        else:
            header[i] == 'column {}'.format(i + 1)
    len_header = len(header)
    assert ([len(cont) == len_header for cont in contents])
    return header, contents


def remove_sequence_number_column(header, contents):
    for i in reversed(range(len(header))):

        ret = [cont[i] for cont in contents]
        if is_squence_number(ret):
            [cont.pop(i) for cont in contents]
            header.pop(i)
    len_header = len(header)
    assert ([len(cont) == len_header for cont in contents])
    return header, contents



def remove_removed_contents(data, contents):
    if len(contents[0])!=len(data[0]):
        # 若data和contents的列数相等，则不需要remove
        col_idx_contents= len(contents[0])-1
        for col_idx_data in reversed(range(len(data[0]))):
            if col_idx_contents < 0:
                [d.pop(col_idx_data) for d in data]
                continue
            col_contents = [item[col_idx_contents] for item in contents]
            col_data = [item[col_idx_data][0] for item in data]
#             print("cont {} {}".format(col_idx_contents, col_contents))
#             print("data {} {}".format(col_idx_data, col_data))
            if col_data != col_contents:
                [d.pop(col_idx_data) for d in data]
            else:
                col_idx_contents -= 1
        assert len(contents[0])==len(data[0])
    return data

def remove_removed_contents(data, contents):
    if len(contents[0])!=len(data[0]):
        # 若data和contents的列数相等，则不需要remove
        col_idx_contents= len(contents[0])-1
        for col_idx_data in reversed(range(len(data[0]))):
            if col_idx_contents < 0:
                [d.pop(col_idx_data) for d in data]
                continue
            col_contents = [item[col_idx_contents] for item in contents]
            col_data = [item[col_idx_data][0] for item in data]
#             print("cont {} {}".format(col_idx_contents, col_contents))
#             print("data {} {}".format(col_idx_data, col_data))
            if col_data != col_contents:
                [d.pop(col_idx_data) for d in data]
            else:
                col_idx_contents -= 1
        assert len(contents[0])==len(data[0])
    return data

