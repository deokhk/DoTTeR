
from typing import Dict, List, Optional

import torch
from dataclasses import dataclass, field

@dataclass
class CLSGCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
    A dictionary of tensors
    """
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    max_csep_index = torch.stack([example['max_csep_index'] for example in batch])
    min_csep_index = torch.stack([example['min_csep_index'] for example in batch])
    batched = {
        'input_ids': input_ids, 
        'attention_mask': attention_mask,
        'max_csep_index': max_csep_index, 
        'min_csep_index': min_csep_index
    }

    if 'median_csep_index' in batch[0].keys():
      median_csep_index = torch.stack([example['median_csep_index'] for example in batch])
      batched.update({'median_csep_index': median_csep_index})

    return batched