from transformers import RobertaModel, RobertaPreTrainedModel
from torch import nn
from typing import List, Optional, Tuple, Union
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import ModelOutput
from dataclasses import dataclass

@dataclass 
class CLSGMinModelOutput(ModelOutput):
    """
    Output type of the CLSGMinModel
    """
    loss: Optional[torch.FloatTensor] = None
    max_logits: torch.FloatTensor = None
    min_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass 
class CLSGMinMaxMedianModelOutput(ModelOutput):
    """
    Output type of the CLSGMinModel
    """
    loss: Optional[torch.FloatTensor] = None
    max_logits: torch.FloatTensor = None
    min_logits: torch.FloatTensor = None
    median_logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class RoBertaCLSGMinMaxModel(RobertaPreTrainedModel):
    """
    Modified model on RobertaForQuestionAnswering
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels, vocab_size):
        super().__init__(config)
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.roberta.resize_token_embeddings(self.vocab_size) 
        self.linear_layer = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_csep_index: Optional[torch.LongTensor] = None,
        min_csep_index: Optional[torch.LongTensor] = None,

        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        max_csep_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the [C_SEP] token with max value in the column for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        min_csep_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the [C_SEP] token with min value in the column for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        sequence_output = outputs[0]

        logits = self.linear_layer(sequence_output)
        max_logits, min_logits = logits.split(1, dim=-1)
        max_logits = max_logits.squeeze(-1).contiguous()
        min_logits = min_logits.squeeze(-1).contiguous()

        total_loss = None
        if max_csep_index is not None and min_csep_index is not None:
            # If we are on multi-GPU, split add a dimension
            if len(max_csep_index.size()) > 1:
                max_csep_index = max_csep_index.squeeze(-1)
            if len(min_csep_index.size()) > 1:
                min_csep_index = min_csep_index.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = max_logits.size(1)
            max_csep_index = max_csep_index.clamp(0, ignored_index)
            min_csep_index = min_csep_index.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(max_logits, max_csep_index)
            end_loss = loss_fct(min_logits, min_csep_index)
            total_loss = (start_loss + end_loss) / 2

        return CLSGMinModelOutput(
            loss = total_loss,
            max_logits=max_logits,
            min_logits=min_logits,
            last_hidden_state =outputs.last_hidden_state ,
            attentions=outputs.attentions,
        )

class RoBertaCLSGMinMaxMedianModel(RobertaPreTrainedModel):
    """
    Modified model on RobertaForQuestionAnswering
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels, vocab_size):
        super().__init__(config)
        self.num_labels = num_labels
        self.vocab_size = vocab_size

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.roberta.resize_token_embeddings(self.vocab_size) 
        self.linear_layer = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        max_csep_index: Optional[torch.LongTensor] = None,
        min_csep_index: Optional[torch.LongTensor] = None,
        median_csep_index: Optional[torch.LongTensor] = None,

        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        max_csep_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the [C_SEP] token with max value in the column for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        min_csep_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the [C_SEP] token with min value in the column for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        median_csep_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the [C_SEP] token with median value in the column for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        sequence_output = outputs[0]

        logits = self.linear_layer(sequence_output)
        max_logits, min_logits, median_logits = logits.split(1, dim=-1)
        max_logits = max_logits.squeeze(-1).contiguous()
        min_logits = min_logits.squeeze(-1).contiguous()
        median_logits = median_logits.squeeze(-1).contiguous()

        total_loss = None
        if max_csep_index is not None and min_csep_index is not None and median_csep_index is not None:
            # If we are on multi-GPU, split add a dimension
            if len(max_csep_index.size()) > 1:
                max_csep_index = max_csep_index.squeeze(-1)
            if len(min_csep_index.size()) > 1:
                min_csep_index = min_csep_index.squeeze(-1)
            if len(median_csep_index.size()) > 1:
                median_csep_index = median_csep_index.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = max_logits.size(1)
            max_csep_index = max_csep_index.clamp(0, ignored_index)
            min_csep_index = min_csep_index.clamp(0, ignored_index)
            median_csep_index = median_csep_index.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(max_logits, max_csep_index)
            end_loss = loss_fct(min_logits, min_csep_index)
            median_loss = loss_fct(median_logits, median_csep_index)
            total_loss = (start_loss + end_loss + median_loss) / 3

        return CLSGMinMaxMedianModelOutput(
            loss = total_loss,
            max_logits=max_logits,
            min_logits=min_logits,
            median_logits=median_logits,
            last_hidden_state =outputs.last_hidden_state ,
            attentions=outputs.attentions,
        )