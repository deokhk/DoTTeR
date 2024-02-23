import sys
sys.path.append('../')
import collections
import logging
import json
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader
from functools import partial

from retrieval.data.encode_datasets import EmDataset, em_collate_bert, em_collate_bert_with_global
from retrieval.models.retriever import CtxEncoder, RobertaCtxEncoder
from retrieval.data.encode_datasets import EmDataset, EmDatasetFilter, EmDatasetMeta, EmDatasetMetaThreeCat, EmDatasetMetaRankThreeCat, EmDatasetMetaColumnGlobalThreeCat, EmDatasetMetaFusionThreeCat
from retrieval.models.retriever import SingleRetriever, SingleEncoder, RobertaSingleEncoder
from retrieval.models.tb_retriever import SingleEncoderThreeCatPool, RobertaSingleEncoderThreeCatPool
from retrieval.models.tb_retriever import SingleEncoderThreeCatAtt, RobertaSingleEncoderThreeCatAtt
from retrieval.models.global_retriever import GlobalSingleEncoderThreeCatPool, RobertaGlobalSingleEncoderThreeCatPool, GlobalColumnSingleEncoderThreeCatPool, RobertaColumnGlobalSingleEncoderThreeCatPool, RobertaFusionSingleEncoderThreeCatPool
from retrieval.config import encode_args
from retrieval.utils.utils import move_to_cuda, load_saved

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    args = encode_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    if not args.predict_file:
        raise ValueError(
            "If `do_predict` is True, then `predict_file` must be specified.")

    assert args.add_special_tokens == False, "For now, only support add_special_tokens == False"

    # select encoding model
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.inject_summary:
        # Make sure to append new tokens to the tokenizer
        if "roberta" in args.model_name:
            summary_token = "madeupword0000"
        elif "bert" in args.model_name: 
            summary_token = "[unused0]"
        else:
            raise ValueError("Model not supported for summary injection")
        num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': [summary_token]})
        assert num_added_toks == 0, "Summary token is not existing in the tokenizer"
        collate_fc = partial(em_collate_bert_with_global, pad_id=tokenizer.pad_token_id)
    else:
        collate_fc = partial(em_collate_bert, pad_id=tokenizer.pad_token_id)

    if "roberta" in args.model_name:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = RobertaSingleEncoderThreeCatAtt(bert_config, args)
                logger.info("Model Using RobertaSingleEncoderThreeCatAtt...")
            else:
                # TODO: current default setting. Support more options in the future
                if args.inject_summary:
                    if args.rate_model_path is not None:
                        column_model_tokenizer = AutoTokenizer.from_pretrained(args.rate_model_path)
                        assert '[C_SEP]' in column_model_tokenizer.additional_special_tokens
                        num_tokens_cm = len(column_model_tokenizer) 
                        if "fusion" in args.injection_scheme:
                            model = RobertaFusionSingleEncoderThreeCatPool(bert_config, num_tokens_cm, args)
                            logger.info("Model Using RobertaFusionSingleEncoderThreeCatPool...")
                        else:
                            model = RobertaColumnGlobalSingleEncoderThreeCatPool(bert_config, num_tokens_cm, args)
                            logger.info("Model Using RobertaColumnGlobalSingleEncoderThreeCatPool...")
                    else:
                        model = RobertaGlobalSingleEncoderThreeCatPool(bert_config, args)
                        logger.info("Model Using RobertaGlobalSingleEncoderThreeCatPool...")
                else:
                    model = RobertaSingleEncoderThreeCatPool(bert_config, args)
                    logger.info("Model Using RobertaSingleEncoderThreeCatPool...")
        else:
            model = RobertaSingleEncoder(bert_config, args)
            logger.info("Model Using RobertaSingleEncoder...")
    else:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = SingleEncoderThreeCatAtt(bert_config, args)
                logger.info("Model Using SingleEncoderThreeCatAtt...")
            else:
                assert "fusion" not in args.inject_scheme, "Currently, we don't support fusion scheme for bert model"
                if args.inject_summary:
                    assert args.rate_model_path is not None, "This configuration is not implemented yet."
                    model = GlobalSingleEncoderThreeCatPool(bert_config, args)
                    logger.info("Model Using GlobalSingleEncoderThreeCatPool...")
                else:
                    model = SingleEncoderThreeCatPool(bert_config, args)
                    logger.info("Model Using SingleEncoderThreeCatPool...")
        else:
            model = SingleEncoder(bert_config, args)
            logger.info("Model Using SingleEncoder...")
    if args.add_special_tokens:
        # special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    # select dataset
    if args.tfidf_filter and args.encode_table:
        eval_dataset = EmDatasetFilter(tokenizer, args.predict_file, args.tfidf_result_file, args.encode_table, args)
        logger.info("Dataset Using EmDatasetFilter...")
    elif args.metadata:
        if args.three_cat:
            # Default selection 
            if args.rank_info:
                eval_dataset = EmDatasetMetaRankThreeCat(tokenizer, args.predict_file, args.encode_table, args)
                logger.info("Dataset Using EmDatasetMetaRankThreeCat...")
            elif args.inject_summary:
                if args.rate_model_path is not None:
                    if "fusion" in args.injection_scheme:
                        eval_dataset = EmDatasetMetaFusionThreeCat(column_model_tokenizer, tokenizer, args.predict_file, args.encode_table, args)
                        logger.info("Dataset Using EmDatasetMetaFusionThreeCat...")
                    else:
                        eval_dataset = EmDatasetMetaColumnGlobalThreeCat(column_model_tokenizer, tokenizer, args.predict_file, args.encode_table, args)
                        logger.info("Dataset Using EmDatasetMetaGlobalThreeCat...")
            else:
                eval_dataset = EmDatasetMetaThreeCat(tokenizer, args.predict_file, args.encode_table, args)
                logger.info("Dataset Using EmDatasetMetaThreeCat...")
        else:
            eval_dataset = EmDatasetMeta(tokenizer, args.predict_file, args.encode_table, args)
            logger.info("Dataset Using EmDatasetMeta...")
    else:
        eval_dataset = EmDataset(tokenizer, args.predict_file, args.encode_table, args)
        logger.info("Dataset Using EmDataset...")

    eval_dataset.processing_data()
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.predict_batch_size,
                                 collate_fn=collate_fc,
                                 pin_memory=True,
                                 num_workers=args.num_workers) 

    assert args.init_checkpoint != ""
    model = load_saved(model, args.init_checkpoint, exact=False)
    model.to(device)


    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # import pdb; pdb.set_trace()

    embeds = predict(model, eval_dataloader)
    logger.info(embeds.size())

    if not os.path.exists(os.path.dirname(args.embed_save_path)):
        os.makedirs(os.path.dirname(args.embed_save_path))
        logger.info("making dir :{}".format(os.path.dirname(args.embed_save_path)))
    logger.info("saving to :{}".format(args.embed_save_path))
    np.save(args.embed_save_path, embeds.cpu().numpy())


def predict(model, eval_dataloader):
    if type(model) == list:
        model = [m.eval() for m in model]
    else:
        model.eval()

    embed_array = []
    # import pdb; pdb.set_trace()
    # logger.info("start from 379200")
    for idx, batch in enumerate(tqdm(eval_dataloader)):
        batch = move_to_cuda(batch)
        with torch.no_grad():
            try:
                results = model(batch)
            except Exception as e:
                logger.info(e)
                # logger.info("Error Batch: {}, instance: {}".format(idx, idx*1600))
                continue
            embed = results['embed'].cpu()
            embed_array.append(embed)

    ## linear combination tuning on dev data
    embed_array = torch.cat(embed_array, dim=0)

    # model.train()
    return embed_array


if __name__ == "__main__":
    main()