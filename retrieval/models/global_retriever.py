"""
Global table info aware retrieval models 
"""


from transformers import AutoModel 
from transformers.models.roberta.modeling_roberta import RobertaAttention, ACT2FN
import torch.nn as nn 
import torch 

import logging
logger = logging.getLogger(__name__)

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from models.tb_retriever import pooling_masked_part
from RATE.models.clsg_model import RoBertaCLSGMinMaxModel, RoBertaCLSGMinMaxMedianModel

class GlobalRetrieverThreeCatPool(nn.Module):
    def __init__(self, config, args):
        super(GlobalRetrieverThreeCatPool, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        
        self.summary_encoder = AutoModel.from_pretrained(args.model_name)
        self.summary_proj = args.summary_proj

        self.injection_scheme = args.injection_scheme # Summary injection option
        self.args = args 

    def forward(self, batch):
        # TODO: summary token index is hard coded. This works currently but maybe need to change it later.
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'],
                                summary_input_ids = batch['c_table_input_ids'], summary_attention_mask=batch['c_table_mask'], summary_token_type_ids=batch['c_table_type_ids'], summary_token_index=6)
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'],
                                summary_input_ids = batch['neg_table_input_ids'], summary_attention_mask=batch['neg_table_mask'], summary_token_type_ids=batch['neg_table_type_ids'], summary_token_index=6)
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}


    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None,
                   
                   summary_input_ids = None,
                   summary_attention_mask = None,
                   summary_token_type_ids = None, 
                   summary_token_index = 6,
                   ):
    
        # First, get the table summary embedding.

        summary_embedding = self.encode_summary(input_ids=summary_input_ids,
                                                attention_mask=summary_attention_mask, 
                                                token_type_ids=summary_token_type_ids)

        if self.injection_scheme == 'embed':
            # We inject the summary embedding to the input embedding
            # Note that below method has only been tested for BERT-family models.
            input_embeddings = self.encoder.base_model.embeddings.word_embeddings(input_ids)
            input_embeddings[:,summary_token_index, :] = summary_embedding 
            hidden_states = self.encoder(inputs_embeds=input_embeddings,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         position_ids=position_ids,
                                         head_mask=head_mask,
                                         output_hidden_states=output_hidden_states)[0]
            cls_rep = hidden_states[:, 0, :]

            # pooled_output = self.dropout(cls_rep[1])
            if self.no_proj:
                part1 = cls_rep
            else:
                part1 = self.project(cls_rep)
            part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
            part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)

            vector = torch.cat([part1, part2, part3], dim=1)

            return vector

        elif self.injection_scheme == "add":
            # We inject the summary embedding by adding with final embedding 
            # Here, the input_ids should not contain the summary token.
            hidden_states = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        output_hidden_states=output_hidden_states)[0]

            cls_rep = hidden_states[:, 0, :]
            if self.no_proj:
                part1 = cls_rep
            else:
                part1 = self.project(cls_rep)
            part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
            part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)


            # Addition 
            part1 = part1 + summary_embedding
            part2 = part2 + summary_embedding
            part3 = part3 + summary_embedding

            vector = torch.cat([part1, part2, part3], dim=1)

            return vector

        else:
            raise ValueError("Invalid summary injection option")



    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_summary(self, input_ids=None,
                       attention_mask=None,
                       token_type_ids=None,
                       ):
        # Get the [CLS] or pooled output of the whole table and 
        # Return it as a summary embedding
        outputs = self.summary_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if not self.args.summary_proj:
            summary_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            summary_embedding = outputs.pooler_output 

        return summary_embedding

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                summary_input_ids = batch['table_input_ids'], summary_attention_mask=batch['table_mask'], summary_token_type_ids=batch['table_type_ids'], summary_token_index=6)
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}



class RobertaGlobalRetrieverThreeCatPool(GlobalRetrieverThreeCatPool):

    def __init__(self,config, args):
        super(RobertaGlobalRetrieverThreeCatPool, self).__init__(config, args)
    
    def forward(self, batch):
        # Here, we don't need to pass type ids 
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'],
                                summary_input_ids = batch['c_table_input_ids'], summary_attention_mask=batch['c_table_mask'], summary_token_index=6)
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'],
                                summary_input_ids = batch['neg_table_input_ids'], summary_attention_mask=batch['neg_table_mask'], summary_token_index=6)
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'],  part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                summary_input_ids = batch['table_input_ids'], summary_attention_mask=batch['table_mask'], summary_token_index=6)
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}


class GlobalSingleEncoderThreeCatPool(nn.Module):
    def __init__(self, config, args):
        super(GlobalSingleEncoderThreeCatPool, self).__init__()
        self.config = config
        self.args = args
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.part_pooling = args.part_pooling

        self.encoder = AutoModel.from_pretrained(args.model_name)
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

        self.summary_encoder = AutoModel.from_pretrained(args.model_name)
        self.summary_proj = args.summary_proj

        self.injection_scheme = args.injection_scheme # Summary injection option
        self.args = args 


    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                    summary_input_ids = batch['table_input_ids'], summary_attention_mask=batch['table_mask'], summary_token_type_ids=batch['table_type_ids'], summary_token_index=6)
                                
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None,

                   summary_input_ids = None,
                   summary_attention_mask = None,
                   summary_token_type_ids = None, 
                   summary_token_index = 6,
                   ):
        # First, get the summary embedding. 

        summary_embedding = self.encode_summary(input_ids=summary_input_ids,
                                                attention_mask=summary_attention_mask, 
                                                token_type_ids=summary_token_type_ids)
    
        if self.injection_scheme == 'embed':
            # We inject the summary embedding to the input embedding
            # Note that below method has only been tested for BERT-family models.
            input_embeddings = self.encoder.base_model.embeddings.word_embeddings(input_ids)
            input_embeddings[:,summary_token_index, :] = summary_embedding 
            hidden_states = self.encoder(inputs_embeds=input_embeddings,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         position_ids=position_ids,
                                         head_mask=head_mask,
                                         output_hidden_states=output_hidden_states)[0]
            cls_rep = hidden_states[:, 0, :]

            # pooled_output = self.dropout(cls_rep[1])
            if self.no_proj:
                part1 = cls_rep
            else:
                part1 = self.project(cls_rep)
            part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
            part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)

            vector = torch.cat([part1, part2, part3], dim=1)

            return vector

        elif self.injection_scheme == "add":
            # We inject the summary embedding by adding with final embedding 
            # Here, the input_ids should not contain the summary token.
            hidden_states = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        output_hidden_states=output_hidden_states)[0]

            cls_rep = hidden_states[:, 0, :]
            if self.no_proj:
                part1 = cls_rep
            else:
                part1 = self.project(cls_rep)
            part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
            part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)


            # Addition 
            part1 = part1 + summary_embedding
            part2 = part2 + summary_embedding
            part3 = part3 + summary_embedding

            vector = torch.cat([part1, part2, part3], dim=1)

            return vector

        else:
            raise ValueError("Invalid summary injection option")

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_summary(self, input_ids=None,
                       attention_mask=None,
                       token_type_ids=None,
                       ):
        # Get the [CLS] or pooled output of the whole table and 
        # Return it as a summary embedding
        outputs = self.summary_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if not self.args.summary_proj:
            summary_embedding = outputs.last_hidden_state[:, 0, :]
        else:
            summary_embedding = outputs.pooler_output 

        return summary_embedding

class RobertaGlobalSingleEncoderThreeCatPool(GlobalSingleEncoderThreeCatPool):

    def __init__(self, config, args):
        super(RobertaGlobalSingleEncoderThreeCatPool, self).__init__(config, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                  summary_input_ids = batch['table_input_ids'], summary_attention_mask = batch['table_mask'], 
                                  summary_token_index=6)
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}



class GlobalColumnSingleEncoderThreeCatPool(nn.Module):
    def __init__(self, config, num_cm_token, args):
        super(GlobalColumnSingleEncoderThreeCatPool, self).__init__()
        self.config = config
        self.args = args
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.part_pooling = args.part_pooling

        self.encoder = AutoModel.from_pretrained(args.model_name)
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

        if args.column_embedding_model_objective == "minmax":
            self.column_summary_model = RoBertaCLSGMinMaxModel.from_pretrained(args.rate_model_path, config=config, num_labels=2, vocab_size=num_cm_token)
        elif args.column_embedding_model_objective == "minmaxmedian":
            self.column_summary_model = RoBertaCLSGMinMaxMedianModel.from_pretrained(args.rate_model_path, config=config, num_labels=3, vocab_size=num_cm_token)
        self.column_summary_model.eval()

        self.summary_proj = args.summary_proj

        self.injection_scheme = args.injection_scheme # Summary injection option
        self.args = args 


    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                column_indices_list=batch['column_token_indices_list'])
                                
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None,

                   column_input_ids_list = None,
                   column_mask_list = None,
                   column_indices_list=None,
                   ):
        # First, get the summary embedding. 
        column_embeddings_list_batch = self.encode_column_embeddings(column_input_ids_list=column_input_ids_list,
                                                            column_mask_list=column_mask_list, 
                                                            column_indices_list=column_indices_list
                                                            )    
        input_embeddings = self.encoder.base_model.embeddings.word_embeddings(input_ids) # [batch_size, seq_len, hidden_size]
        for bidx, (column_embeddings_list, cur_input_id) in enumerate(zip(column_embeddings_list_batch, input_ids)):
            # Get the index of [COL1]  ~ [COLN] token
            # These are encoded with input_id = 50261 / which is "madeupword0000"
            column_token_indices = torch.where(cur_input_id == 50261)[0]
            for column_embedding, col_tok_index in zip(column_embeddings_list, column_token_indices):
                try:
                    # Inject the column embedding to the corresponding position in the input embedding
                    input_embeddings[bidx, col_tok_index, :] = column_embedding
                except IndexError:
                    breakpoint()
        
        hidden_states = self.encoder(inputs_embeds=input_embeddings,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]

        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)

        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_column_embeddings(self, column_input_ids_list=None,
                            column_mask_list=None, 
                            column_indices_list=None,
                            ):
        
        column_embeddings_list_batch = []
        # We need to know the index of each column embedding in the input_ids
        for column_input_ids, column_mask, column_indices in zip(column_input_ids_list, column_mask_list, column_indices_list):
            # We need to truncate the padding tokens
            is_padded = torch.any(column_indices == -1).item()
            if is_padded:
                num_header = torch.where(column_indices == -1)[0][0].item()
            else:
                num_header = column_indices.shape[0]
            column_input_ids_unpadded = column_input_ids[:num_header, :]
            column_mask_unpadded = column_mask[:num_header, :]
            column_indices_unpadded = column_indices[:num_header]
            out = self.column_summary_model(column_input_ids_unpadded, column_mask_unpadded)
            column_embeddings_list = []
            for idx, csep_index in enumerate(column_indices_unpadded):
                # column_index == index of "[C_SEP]" token corresponds to the row 
                last_rep = out.last_hidden_state[idx, csep_index, :] # (col_num, seq_len, dim), where col_num is the number of columns in a single row 
                column_embeddings_list.append(last_rep)
            column_embeddings_list_batch.append(column_embeddings_list)
        return column_embeddings_list_batch



class RobertaColumnGlobalSingleEncoderThreeCatPool(GlobalColumnSingleEncoderThreeCatPool):

    def __init__(self, config, num_cm_token, args):
        super(RobertaColumnGlobalSingleEncoderThreeCatPool, self).__init__(config, num_cm_token, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                  column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                  column_indices_list=batch['column_token_indices_list'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}



class GlobalFusionSingleEncoderThreeCatPool(nn.Module):

    def __init__(self, config, num_cm_token, args):
        super(GlobalFusionSingleEncoderThreeCatPool, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        self.fusion_layer = FusionLayer(config)
        if args.column_embedding_model_objective == "minmax":
            self.column_summary_model = RoBertaCLSGMinMaxModel.from_pretrained(args.rate_model_path, config=config, num_labels=2, vocab_size=num_cm_token)
        elif args.column_embedding_model_objective == "minmaxmedian":
            self.column_summary_model = RoBertaCLSGMinMaxMedianModel.from_pretrained(args.rate_model_path, config=config, num_labels=3, vocab_size=num_cm_token)
        self.column_summary_model.eval()
        
        self.summary_proj = args.summary_proj


        self.args = args 

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                        column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                        column_indices_list=batch['column_token_indices_list'], column_categories=batch['column_categories'],
                                        value_mask=batch["value_mask"])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None,
                   
                   column_input_ids_list = None,
                   column_mask_list = None,
                   column_indices_list=None,
                   column_categories=None,

                   value_mask=None,
                   ):
    
        # First, get the column embeddings list.
        # This is list of column embeddings, where length of list is equal to the batch size.

        column_embeddings_list_batch = self.encode_column_embeddings(column_input_ids_list=column_input_ids_list,
                                                            column_mask_list=column_mask_list, 
                                                            column_indices_list=column_indices_list
                                                            )    
        
        hidden_states = self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    output_hidden_states=output_hidden_states)[0]
        
        hidden_states = self.fusion_layer(hidden_states, column_embeddings_list_batch, column_categories, value_mask)
        cls_rep = hidden_states[:, 0, :]

        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector



    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)

        vector = torch.cat([part1, part2, part3], dim=1)
        return vector


    def encode_column_embeddings(self, column_input_ids_list=None,
                            column_mask_list=None, 
                            column_indices_list=None,
                            ):
        # This is the retriever we are using!
        column_embeddings_list_batch = []
        # We need to know the index of each column embedding in the input_ids
        for column_input_ids, column_mask, column_indices in zip(column_input_ids_list, column_mask_list, column_indices_list):
            # We need to truncate the padding tokens
            is_padded = torch.any(column_indices == -1).item()
            if is_padded:
                num_header = torch.where(column_indices == -1)[0][0].item()
            else:
                num_header = column_indices.shape[0]
            column_input_ids_unpadded = column_input_ids[:num_header, :]
            column_mask_unpadded = column_mask[:num_header, :]
            column_indices_unpadded = column_indices[:num_header]

            out = self.column_summary_model(column_input_ids_unpadded, column_mask_unpadded)
            column_embeddings_list = []
            for idx, csep_index in enumerate(column_indices_unpadded):
                # column_index == index of "[C_SEP]" token corresponds to the row 
                last_rep = out.last_hidden_state[idx, csep_index, :] # (col_num, seq_len, dim), where col_num is the number of columns in a single row 
                column_embeddings_list.append(last_rep)
            column_embeddings_list_batch.append(column_embeddings_list)
        return column_embeddings_list_batch



class RobertaFusionSingleEncoderThreeCatPool(GlobalFusionSingleEncoderThreeCatPool):

    def __init__(self, config, num_cm_token, args):
        super(RobertaFusionSingleEncoderThreeCatPool, self).__init__(config, num_cm_token, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                column_indices_list=batch['column_token_indices_list'], column_categories=batch['column_categories'],
                                value_mask=batch["value_mask"])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}



class GlobalRetrieverColumnThreeCatPool(nn.Module):
    """
    Global retriever, that embed column-level information to the tokens.
    
    """
    def __init__(self, config, num_cm_token, args):
        super(GlobalRetrieverColumnThreeCatPool, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        
        if args.column_embedding_model_objective == "minmax":
            self.column_summary_model = RoBertaCLSGMinMaxModel.from_pretrained(args.rate_model_path, config=config, num_labels=2, vocab_size=num_cm_token)
        elif args.column_embedding_model_objective == "minmaxmedian":
            self.column_summary_model = RoBertaCLSGMinMaxMedianModel.from_pretrained(args.rate_model_path, config=config, num_labels=3, vocab_size=num_cm_token)
        self.column_summary_model.eval()
        
        self.summary_proj = args.summary_proj

        self.args = args 

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'],
                                column_input_ids_list = batch['c_column_input_ids_list'], column_mask_list=batch['c_column_mask_list'],
                                column_indices_list=batch['c_column_token_indices_list'])
        
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'],
                                column_input_ids_list = batch['neg_column_input_ids_list'], column_mask_list=batch['neg_column_mask_list'],
                                column_indices_list=batch['neg_column_token_indices_list'])
        
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}


    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None,
                   
                   column_input_ids_list = None,
                   column_mask_list = None,
                   column_indices_list=None,
                   ):
    
        # First, get the column embeddings list.
        # This is list of column embeddings, where length of list is equal to the batch size.

        column_embeddings_list_batch = self.encode_column_embeddings(column_input_ids_list=column_input_ids_list,
                                                            column_mask_list=column_mask_list, 
                                                            column_indices_list=column_indices_list
                                                            )    
        # We inject the summary embedding to the input embedding
        # Note that below method has only been tested for BERT-family models.
        input_embeddings = self.encoder.base_model.embeddings.word_embeddings(input_ids) # [batch_size, seq_len, hidden_size]
        for bidx, (column_embeddings_list, cur_input_id) in enumerate(zip(column_embeddings_list_batch, input_ids)):
            # Get the index of [COL1]  ~ [COLN] token
            # These are encoded with input_id = 50261 / which is "madeupword0000"
            column_token_indices = torch.where(cur_input_id == 50261)[0]
            for column_embedding, col_tok_index in zip(column_embeddings_list, column_token_indices):
                try:
                    # Inject the column embedding to the corresponding position in the input embedding
                    input_embeddings[bidx, col_tok_index, :] = column_embedding
                except IndexError:
                    breakpoint()
        
        hidden_states = self.encoder(inputs_embeds=input_embeddings,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]

        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector



    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)

        vector = torch.cat([part1, part2, part3], dim=1)
        return vector


    def encode_column_embeddings(self, column_input_ids_list=None,
                            column_mask_list=None, 
                            column_indices_list=None,
                            ):
        # This is the retriever we are using!
        column_embeddings_list_batch = []
        # We need to know the index of each column embedding in the input_ids
        for column_input_ids, column_mask, column_indices in zip(column_input_ids_list, column_mask_list, column_indices_list):
            # We need to truncate the padding tokens
            is_padded = torch.any(column_indices == -1).item()
            if is_padded:
                num_header = torch.where(column_indices == -1)[0][0].item()
            else:
                num_header = column_indices.shape[0]
            column_input_ids_unpadded = column_input_ids[:num_header, :]
            column_mask_unpadded = column_mask[:num_header, :]
            column_indices_unpadded = column_indices[:num_header]

            out = self.column_summary_model(column_input_ids_unpadded, column_mask_unpadded)
            column_embeddings_list = []
            for idx, csep_index in enumerate(column_indices_unpadded):
                # column_index == index of "[C_SEP]" token corresponds to the row 
                last_rep = out.last_hidden_state[idx, csep_index, :] # (col_num, seq_len, dim), where col_num is the number of columns in a single row 
                column_embeddings_list.append(last_rep)
            column_embeddings_list_batch.append(column_embeddings_list)
        return column_embeddings_list_batch


    def evaluate_encode_tb(self, batch):        
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                        column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                        column_indices_list=batch['column_token_indices_list'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):

        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}


class RobertaGlobalRetrieverColumnThreeCatPool(GlobalRetrieverColumnThreeCatPool):

    def __init__(self,config, num_cm_token, args):
        super(RobertaGlobalRetrieverColumnThreeCatPool, self).__init__(config, num_cm_token, args)
    
    def forward(self, batch):
        # Here, we don't need to pass type ids 
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'],
                                column_input_ids_list = batch['c_column_input_ids_list'], column_mask_list=batch['c_column_mask_list'],
                                column_indices_list=batch['c_column_token_indices_list'])
        
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'],
                                column_input_ids_list = batch['neg_column_input_ids_list'], column_mask_list=batch['neg_column_mask_list'],
                                column_indices_list=batch['neg_column_token_indices_list'])
        
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                        column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                        column_indices_list=batch['column_token_indices_list'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}








class FusionLayer(nn.Module):
    """
    Fusion layer, used for mix token and column embeddings.
    """
    def __init__(self, config):
        # linear layer 
        super(FusionLayer, self).__init__()
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_col = nn.Linear(config.hidden_size, config.hidden_size) # TODO: From now, the output dimension of column embedding is same as the token embedding. Make it more robust?

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act
        # attention layer, followed by output layer
        self.attention_layer = RobertaAttention(config)


    def forward(self, hidden_states, column_embeddings_list_batch, column_categories, value_mask):
        """
            hidden_states (tensor) : [batch_size, max_seq_length, config.hidden_size]
            column_embeddings_list_batch (List[List]) : [batch_size, num(column), config.hidden_size]
            column_categories (tensor) : [batch_size, num(column)]
            value_mask: (tensor) : [batch_size, max_seq_length]

        """        
        
        # value_mask = [batch_size, max_seq_length]. If the token is a value token, then index of column corresopnding to value + 1. Otherwise, 0.
        # We take motivation from implementation of ERNIE 
        hidden_states_ = self.dense(hidden_states) # [batch_size, max_seq_length,]

        hidden_states_col = torch.zeros_like(hidden_states)
        # Compute hidden_states_col
        for bidx, (column_embeddings_list, vmask, column_category) in enumerate(zip(column_embeddings_list_batch, value_mask, column_categories)):
            map = dict()
            for idx, (column_embedding, category) in enumerate(zip(column_embeddings_list, column_category)):
                if category == 1:
                    # nuemeric or date type
                    map[idx+1] = column_embedding
                elif category == 0:
                    map[idx+1] = torch.zeros_like(column_embedding)
            map[0] = torch.zeros_like(column_embeddings_list[0]) # If the token is not a value token, then we use zero vector.
            hidden_states_col[bidx, :, :] = torch.stack([map[idx.item()] for idx in vmask])
        hidden_states_col_ = self.dense_col(hidden_states_col)

        hidden_states = self.act_fn(hidden_states_ + hidden_states_col_)
        hidden_states = self.attention_layer(hidden_states)[0]
        return hidden_states 

class FusionRetrieverColumnThreeCatPool(nn.Module):
    """
    Global retriever, that fuses column-level information to the tokens.
    
    """
    def __init__(self, config, num_cm_token, args):
        super(FusionRetrieverColumnThreeCatPool, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        self.fusion_layer = FusionLayer(config)
        if args.column_embedding_model_objective == "minmax":
            self.column_summary_model = RoBertaCLSGMinMaxModel.from_pretrained(args.rate_model_path, config=config, num_labels=2, vocab_size=num_cm_token)
        elif args.column_embedding_model_objective == "minmaxmedian":
            self.column_summary_model = RoBertaCLSGMinMaxMedianModel.from_pretrained(args.rate_model_path, config=config, num_labels=3, vocab_size=num_cm_token)
        self.column_summary_model.eval()
        
        self.summary_proj = args.summary_proj


        self.args = args 

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'],
                                column_input_ids_list = batch['c_column_input_ids_list'], column_mask_list=batch['c_column_mask_list'],
                                column_indices_list=batch['c_column_token_indices_list'], column_categories=batch['c_column_categories'],
                                value_mask=batch["c_value_mask"])
        
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'],
                                column_input_ids_list = batch['neg_column_input_ids_list'], column_mask_list=batch['neg_column_mask_list'],
                                column_indices_list=batch['neg_column_token_indices_list'], column_categories=batch['neg_column_categories'],
                                value_mask=batch["neg_value_mask"])
        
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}


    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None,
                   
                   column_input_ids_list = None,
                   column_mask_list = None,
                   column_indices_list=None,
                   column_categories=None,

                   value_mask=None,
                   ):
    
        # First, get the column embeddings list.
        # This is list of column embeddings, where length of list is equal to the batch size.

        column_embeddings_list_batch = self.encode_column_embeddings(column_input_ids_list=column_input_ids_list,
                                                            column_mask_list=column_mask_list, 
                                                            column_indices_list=column_indices_list
                                                            )    
        
        hidden_states = self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    output_hidden_states=output_hidden_states)[0]
        
        hidden_states = self.fusion_layer(hidden_states, column_embeddings_list_batch, column_categories, value_mask)
        cls_rep = hidden_states[:, 0, :]

        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector



    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        if len(part2.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part2 = part2.unsqueeze(0)
        if len(part3.shape) == 1:
            # If it is a 1d tensor, than we add batch dimension
            part3 = part3.unsqueeze(0)

        vector = torch.cat([part1, part2, part3], dim=1)
        return vector


    def encode_column_embeddings(self, column_input_ids_list=None,
                            column_mask_list=None, 
                            column_indices_list=None,
                            ):
        # This is the retriever we are using!
        column_embeddings_list_batch = []
        # We need to know the index of each column embedding in the input_ids
        for column_input_ids, column_mask, column_indices in zip(column_input_ids_list, column_mask_list, column_indices_list):
            # We need to truncate the padding tokens
            is_padded = torch.any(column_indices == -1).item()
            if is_padded:
                num_header = torch.where(column_indices == -1)[0][0].item()
            else:
                num_header = column_indices.shape[0]
            column_input_ids_unpadded = column_input_ids[:num_header, :]
            column_mask_unpadded = column_mask[:num_header, :]
            column_indices_unpadded = column_indices[:num_header]

            out = self.column_summary_model(column_input_ids_unpadded, column_mask_unpadded)
            column_embeddings_list = []
            for idx, csep_index in enumerate(column_indices_unpadded):
                # column_index == index of "[C_SEP]" token corresponds to the row 
                last_rep = out.last_hidden_state[idx, csep_index, :] # (col_num, seq_len, dim), where col_num is the number of columns in a single row 
                column_embeddings_list.append(last_rep)
            column_embeddings_list_batch.append(column_embeddings_list)
        return column_embeddings_list_batch


    def evaluate_encode_tb(self, batch):        
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                        column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                        column_indices_list=batch['column_token_indices_list'], column_categories=batch['column_categories'],
                                        value_mask=batch["value_mask"])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):

        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}




class RobertaFusionRetrieverColumnThreeCatPool(FusionRetrieverColumnThreeCatPool):

    def __init__(self,config, num_cm_token, args):
        super(RobertaFusionRetrieverColumnThreeCatPool, self).__init__(config, num_cm_token, args)
    
    def forward(self, batch):
        # Here, we don't need to pass type ids 
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'],
                                column_input_ids_list = batch['c_column_input_ids_list'], column_mask_list=batch['c_column_mask_list'],
                                column_indices_list=batch['c_column_token_indices_list'], column_categories=batch['c_column_categories'],
                                value_mask=batch["c_value_mask"])
        
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'],
                                column_input_ids_list = batch['neg_column_input_ids_list'], column_mask_list=batch['neg_column_mask_list'],
                                column_indices_list=batch['neg_column_token_indices_list'], column_categories=batch['neg_column_categories'],
                                value_mask=batch["neg_value_mask"])
        
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'],
                                        column_input_ids_list = batch['column_input_ids_list'], column_mask_list=batch['column_mask_list'],
                                        column_indices_list=batch['column_token_indices_list'], column_categories=batch['column_categories'],
                                        value_mask=batch["value_mask"])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}
