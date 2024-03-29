import collections
import re
import string
import torch 

def collate_tokens(values, pad_idx,input_type = '1d', eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""    
    
    if len(values[0].size()) > 1:
        if input_type=='1d':
            values = [v.view(-1) for v in values]
        elif input_type=='2d':
            values = [v.view(-1, v.size(2)) for v in values]
            res_dim = values[0].size(1)
    size = max(v.size(0) for v in values)
    if input_type=='1d':
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif input_type=='2d':
        res = values[0].new(len(values),size,res_dim).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):][:] if left_pad else res[i][:len(v)][:])

    return res


def collate_tokens_to_3d(values, pad_idx):
    """Convert a list of 2d tensors into a padded 3d tensor.
    Shape of 2d tensor : [num_header, seq_len]
    Output: [batch_size, num_header, seq_len]
    (Assume seq_len is eqaul for all the tensors in the batch)
    
    """
    max_num_header = max(v.size(0) for v in values)
    seq_len = values[0].size(1)
    padded_list = []
    for v in values:
        target = torch.full((max_num_header, seq_len), pad_idx)
        target[:v.size(0), :] = v
        padded_list.append(target)
    res = torch.stack(padded_list, dim=0)
    return res 

    

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
