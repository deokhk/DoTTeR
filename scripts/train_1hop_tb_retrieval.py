import logging
import os
import sys
import random
from tqdm import tqdm
from datetime import date
import numpy as np
import wandb 
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.optim import Adam
from functools import partial

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append('../')
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          BertConfig, BertModel, BertTokenizer,
                          XLNetConfig, XLNetModel, XLNetTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          ElectraConfig, ElectraModel, ElectraTokenizer,
                          AlbertConfig, AlbertModel, AlbertTokenizer,
                          LongformerConfig, LongformerModel, LongformerTokenizer)

from retrieval.data.tr_dataset import TRDatasetNega, TRDatasetNegaMeta, TRDatasetNegaWithMetaRank, tb_collate
from retrieval.data.tr_dataset import TRDatasetNegaMetaThreeCat, TRDatasetNegaWithMetaRankThreeCat, TRDatasetNegaWithGlobalMetaThreecat, TRDatasetNegaWithGlobalColumnMetaThreecat, TRDatasetNegaWithFusionColumnMetaThreecat
from retrieval.utils.utils import move_to_cuda, AverageMeter, load_saved
from retrieval.config import train_args
from retrieval.criterions import loss_single
from retrieval.models.retriever import SingleRetriever, RobertaSingleRetriever
from retrieval.models.tb_retriever import SingleRetrieverThreeCatPool, RobertaSingleRetrieverThreeCatPool
from retrieval.models.tb_retriever import SingleRetrieverThreeCatAtt, RobertaSingleRetrieverThreeCatAtt
from retrieval.models.global_retriever import GlobalRetrieverThreeCatPool, RobertaGlobalRetrieverThreeCatPool, RobertaGlobalRetrieverColumnThreeCatPool, RobertaFusionRetrieverColumnThreeCatPool

def set_seed(seed=666):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = train_args()

    run = None
    if args.wandb_log:
        wandb_name = args.exp_name
        run = wandb.init(project="rank_retrieval", name=wandb_name, config=args)

    # date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-{args.model_name}"
    # model_name += f"-k{args.k}" if args.momentum else ""

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"output directory {args.output_dir} already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)
    os.system("touch {}".format(os.path.join(args.output_dir, model_name)))

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, n_gpu, bool(args.local_rank != -1))

    set_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    ################## MODEL SELECTION ####################################
    bert_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(tb_collate, pad_id=tokenizer.pad_token_id)

    resized_token_len=None


    if args.add_special_tokens and args.init_checkpoint == "":
        # We only add special tokens if we're not loading from a pre-trained model
        # Due to the size mismatch
        if args.rank_info and args.rank_scheme == "ctx_minmax":
            special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]", "[MIN]", "[MAX]"]}
        elif args.summary_info:
            # Currently, we don't assume to use summary info with rank info 
            special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]", "[SUMMARY]"]}
        else:
            special_tokens_dict = {'additional_special_tokens': ["[HEADER]", "[PASSAGE]","[SEP]", "[TB]","[DATA]","[TITLE]","[SECTITLE]"]}
        
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        resized_token_len = (len(tokenizer))
        logger.info("Add Special Tokens: {}".format(special_tokens_dict["additional_special_tokens"]))


    if args.inject_summary and args.init_checkpoint != "":
        # This is hacky logic to inject summary token into the model and load pretrained weights simultaneously
        if "roberta" in args.model_name:
            summary_token = "madeupword0000"
        elif "bert" in args.model_name: 
            summary_token = "[unused0]"
        else:
            raise ValueError("Model not supported for summary injection")
        num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': [summary_token]})
        assert num_added_toks == 0, "Summary token is not existing in the tokenizer"


    column_model_tokenizer = None 
    if "roberta" in args.model_name or "longformer" in args.model_name:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = RobertaSingleRetrieverThreeCatAtt(bert_config, args, resized_token_len)
                logger.info("Model Using RobertaSingleRetrieverThreeCatPool...")
            else:
                # Current default setting 
                # TODO: make this more flexible 
                if args.inject_summary:
                    if args.injection_scheme != "embed" or args.injection_scheme != "add":
                        assert args.rate_model_path is not None, "Need to specify rate model path for column-level summary injection"
                    
                    if args.rate_model_path is not None:
                        # Then we need to load another tokenizer for column-level summary generation
                        column_model_tokenizer = AutoTokenizer.from_pretrained(args.rate_model_path)
                        assert '[C_SEP]' in column_model_tokenizer.additional_special_tokens
                        num_tokens_cm = len(column_model_tokenizer) 
                        if "fusion" in args.injection_scheme:
                            model = RobertaFusionRetrieverColumnThreeCatPool(bert_config, num_tokens_cm, args)
                            logger.info("Inject summary using column-specific embedding model, fusion scheme...")
                            logger.info("Model Using RobertaFusionRetrieverColumnThreeCatPool...")
                        else:
                            model = RobertaGlobalRetrieverColumnThreeCatPool(bert_config, num_tokens_cm, args)
                            logger.info("Inject summary using column-specific embedding model...")
                            logger.info("Model Using RobertaGlobalRetrieverColumnThreeCatPool...")

                        # Freeze the summary embedding model
                        for name, p in model.named_parameters():
                            if "column_summary_model" in name:
                                p.requires_grad = False
                        logger.info("Training with frozen weight of column summary model.")

                        # Print number of trainable parameters
                        logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
                    else:
                        model = RobertaGlobalRetrieverThreeCatPool(bert_config, args)
                        logger.info("Model Using RobertaGlobalRetrieverThreeCatPool...")
                else:
                    model = RobertaSingleRetrieverThreeCatPool(bert_config, args, resized_token_len)
                    logger.info("Model Using RobertaSingleRetrieverThreeCatPool...")
        else:
            model = RobertaSingleRetriever(bert_config, args, resized_token_len)
            logger.info("Model Using RobertaSingleRetriever...")
    else:
        if args.three_cat:
            if args.part_pooling[:3] == 'att':
                model = SingleRetrieverThreeCatAtt(bert_config, args)
                logger.info("Model Using SingleRetrieverThreeCatPool...")
            else:
                # Current default setting 
                # TODO: make this more flexible 
                assert "fusion" not in args.inject_scheme, "Currently, we don't support fusion scheme for bert model"
                if args.inject_summary:
                    model = GlobalRetrieverThreeCatPool(bert_config, args)
                    logger.info("Model Using GlobalRetrieverThreeCatPool...")
                else:
                    model = SingleRetrieverThreeCatPool(bert_config, args, resized_token_len)
                    logger.info("Model Using SingleRetrieverThreeCatPool...")
        else:
            model = SingleRetriever(bert_config, args, resized_token_len)
            logger.info("Model Using SingleRetriever...")

    if args.add_special_tokens and args.init_checkpoint == "":
        assert isinstance(model, SingleRetrieverThreeCatPool), "Currently we only support SingleRetrieverThreeCatPool family to add special tokens"

    if args.do_train and args.max_c_len > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, bert_config.max_position_embeddings))

    ################## EVAL DATASET SELECTION####################################
    if args.metadata:
        if args.three_cat:
            if args.rank_info:
                eval_dataset = TRDatasetNegaWithMetaRankThreeCat(tokenizer, args.predict_file, args)
                logger.info("Eval Dataset Using TRDatasetNegaMetaRankThreeCat...")
            elif args.inject_summary:
                if args.rate_model_path is not None:
                    if "fusion" in args.injection_scheme:
                        eval_dataset = TRDatasetNegaWithFusionColumnMetaThreecat(column_model_tokenizer, tokenizer, args.predict_file, args)
                        logger.info("Eval Dataset Using TRDatasetNegaWithFusionColumnMetaThreecat...")
                    else:
                        eval_dataset = TRDatasetNegaWithGlobalColumnMetaThreecat(column_model_tokenizer, tokenizer, args.predict_file, args)
                        logger.info("Eval Dataset Using TRDatasetNegaWithCGlobalColumnMetaThreecat...")
                else:
                    eval_dataset = TRDatasetNegaWithGlobalMetaThreecat(tokenizer, args.predict_file, args)
                    logger.info("Eval Dataset Using TRDatasetNegaWithGlobalMetaThreecat...")
            else:
                eval_dataset = TRDatasetNegaMetaThreeCat(tokenizer, args.predict_file, args)
                logger.info("Eval Dataset Using TRDatasetNegaMetaThreeCat...")
        else:
            if args.rank_info:
                eval_dataset = TRDatasetNegaWithMetaRank(tokenizer, args.predict_file, args)
                logger.info("Eval Dataset Using TRDatasetNegaWithMetaRank...")
            else:
                eval_dataset = TRDatasetNegaMeta(tokenizer, args.predict_file, args)
                logger.info("Eval Dataset Using TRDatasetNegaMeta...")
    else:
        eval_dataset = TRDatasetNega(tokenizer, args.predict_file, args)
        logger.info("Eval Dataset Using TRDatasetNega...")
    eval_dataset.processing_data()

    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=args.predict_batch_size,
                                 collate_fn=collate_fc,
                                 pin_memory=True,
                                 num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        model = load_saved(model, args.init_checkpoint,)

    model.to(device)
    # logger.info(model)
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    if args.do_train:
        # args.train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train()

        ################## TRAIN DATASET SELECTION####################################
        if args.metadata:
            if args.three_cat:
                if args.rank_info:
                    train_dataset = TRDatasetNegaWithMetaRankThreeCat(tokenizer, args.train_file, args, train=True)
                    logger.info("Training Dataset Using TRDatasetNegaMetaRankThreeCat...")
                elif args.inject_summary:
                    if args.rate_model_path is not None:
                        if "fusion" in args.injection_scheme:
                            train_dataset = TRDatasetNegaWithFusionColumnMetaThreecat(column_model_tokenizer, tokenizer, args.train_file, args, train=True)
                            logger.info("Training Dataset Using TRDatasetNegaWithFusionColumnMetaThreecat...")
                        else:
                            train_dataset = TRDatasetNegaWithGlobalColumnMetaThreecat(column_model_tokenizer, tokenizer, args.train_file, args, train=True)
                            logger.info("Training Dataset Using TRDatasetNegaWithGlobalColumnMetaThreecat...")
                    else:
                        train_dataset = TRDatasetNegaWithGlobalMetaThreecat(tokenizer, args.train_file, args, train=True)
                        logger.info("Training Dataset Using TRDatasetNegaWithGlobalMetaThreecat...")
                else:
                    train_dataset = TRDatasetNegaMetaThreeCat(tokenizer, args.train_file, args, train=True)
                    logger.info("Training Dataset Using TRDatasetNegaMetaThreeCat...")
            else:
                if args.rank_info:
                    train_dataset = TRDatasetNegaWithMetaRank(tokenizer, args.train_file, args, train=True)
                    logger.info("Training Dataset Using TRDatasetNegaWithMetaRank...")
                else:
                    train_dataset = TRDatasetNegaMeta(tokenizer, args.train_file, args, train=True)
                    logger.info("Training Dataset Using TRDatasetNegaMeta...")
        else:
            train_dataset = TRDatasetNega(tokenizer, args.train_file, args, train=True)
            logger.info("Training Dataset Using TRDatasetNega...")
        train_dataset.processing_data()
        if args.data_augmentation:
            logger.info("==== Add Augment Data ====")
            train_dataset.add_augment_data(args.augment_file)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True,
                                      collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True) 

        logger.info("main: ")
        logger.info("length: dataset: {}, dataloader: {}".format(len(train_dataset), len(train_dataloader)))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        warmup_steps = t_total * args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)
        model.zero_grad()

        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):

            for batch in tqdm(train_dataloader):
                batch_step += 1
                batch = move_to_cuda(batch)
                model.train()
                loss = loss_single(model, batch)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                train_loss_meter.update(loss.item())

                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if run is not None:
                        run.log({'batch_train_loss': loss.item(), 'smoothed_train_loss': train_loss_meter.avg})

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        mrr = predict(args, model, eval_dataloader, device, logger)
                        logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                        global_step, train_loss_meter.avg, mrr * 100, epoch))

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                                        (best_mrr * 100, mrr * 100, epoch))
                            logger.info("Saving to {}".format(os.path.join(args.output_dir, "checkpoint_best.pt")))
                            torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrr = predict(args, model, eval_dataloader, device, logger)
            logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, mrr * 100, epoch))
            if run is not None:
                run.log({'dev_mrr':mrr * 100})
            if best_mrr < mrr:
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_last.pt"))
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                            (best_mrr * 100, mrr * 100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                model = model.to(device)
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        output_dir = args.output_dir
        model.load_state_dict(torch.load(os.path.join(output_dir, 'checkpoint_best.pt')))
        acc = predict(args, model, eval_dataloader, device, logger)
        logger.info(f"test performance {acc}")


def predict(args, model, eval_dataloader, device, logger):
    model.eval()
    num_correct = 0
    num_total = 0.0
    rrs = []  # reciprocal rank
    for batch in tqdm(eval_dataloader):
        batch_to_feed = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(batch_to_feed)

            q = outputs['q']
            c = outputs['c']
            neg_c = outputs['neg_c']

            product_in_batch = torch.mm(q, c.t())
            product_neg = (q * neg_c).sum(-1).unsqueeze(1)
            product = torch.cat([product_in_batch, product_neg], dim=-1).cpu().detach()

            target = torch.arange(product.size(0)).to(product.device)
            ranked = product.argsort(dim=1, descending=True)
            prediction = product.argmax(-1)

            # MRR
            idx2rank = ranked.argsort(dim=1)
            for idx, t in enumerate(target.tolist()):
                rrs.append(1 / (idx2rank[idx][t].item() + 1))

            pred_res = prediction == target
            num_total += pred_res.size(0)
            num_correct += pred_res.sum(0)

    acc = num_correct / num_total
    mrr = np.mean(rrs)
    logger.info(f"evaluated {num_total} examples...")
    logger.info(f"avg. Acc: {acc}")
    logger.info(f'MRR: {mrr}')
    model.train()
    return mrr


if __name__ == "__main__":
    main()