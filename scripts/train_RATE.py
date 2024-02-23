import argparse 
import os 
import logging 
import sys 
import torch 
import numpy as np
import wandb 

from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_scheduler,
    set_seed
)
from tqdm.auto import tqdm

from torch.utils.data import DataLoader
from torch.optim import AdamW 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RATE.data.sg_dataset import CLSummaryGenDataset
from RATE.models.clsg_model import RoBertaCLSGMinMaxModel, RoBertaCLSGMinMaxMedianModel
from RATE.data.sg_collator import CLSGCollator

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def main(args):

    wandb.login()
    run_name = "RATE_{}_{}_{}".format(args.model_name, args.objective,args.batch_size)
    run = wandb.init(
        # Set the project where this run will be logged
        project="RATE",
        # Track hyperparameters and run metadata
        config={
            "train_steps": args.num_train_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "model_name": args.model_name,
            "objective": args.objective,
        },
        name=run_name
    )

    output_dir = args.output_dir 
    batch_size = args.batch_size
    evaluation_batch_size = args.evaluation_batch_size

    num_gpu_available = torch.cuda.device_count()
    num_workers = int(4 * num_gpu_available)

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    special_tokens_dict = {'additional_special_tokens': ['[C_SEP]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    tokenized_dataset = CLSummaryGenDataset(tokenizer=tokenizer, data_path=args.wikitable_path, args=args)
    dataset_size = len(tokenized_dataset)
    
    val_size = min(int(0.1 * dataset_size), 10000)
    train_size = dataset_size-val_size 

    train_dataset = torch.utils.data.Subset(tokenized_dataset, range(train_size))
    eval_dataset = torch.utils.data.Subset(tokenized_dataset, range(train_size, train_size + val_size))

    logger.info("Train dataset size: {}".format(len(train_dataset)))
    logger.info("Validation dataset size: {}".format(len(eval_dataset)))

    data_collator = CLSGCollator()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=num_workers)
    eval_dataloader = DataLoader(eval_dataset, batch_size=evaluation_batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers)


    logger.info(f"Loading model with {args.model_name} encoder")

    config = AutoConfig.from_pretrained(args.model_name)
    if args.objective == "minmax":
        model = RoBertaCLSGMinMaxModel(config, 2, len(tokenizer))
    elif args.objective == "minmaxmedian":
        model = RoBertaCLSGMinMaxMedianModel(config, 3, len(tokenizer))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        logger.info("Using {} GPUs".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.num_train_steps,
    )

    progress_bar = tqdm(range(args.num_train_steps), desc="Training..")

    # We calculate the number of epochs to satisfy the number of training steps 
    num_epochs = int(args.num_train_steps / len(train_dataloader)) + 1 

    global_step = 0
    best_accuracy = 0
        # Training
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            if args.objective == "minmax":
                outputs = model(input_ids=batch["input_ids"].to(device), 
                                attention_mask=batch["attention_mask"].to(device), 
                                max_csep_index=batch["max_csep_index"].to(device), 
                                min_csep_index=batch["min_csep_index"].to(device)
                            )
                loss = outputs.loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step +=1 
                
                if global_step % args.logging_steps == 0:
                    # Logging 
                    wandb.log({"loss":loss})

                if global_step % args.evaluation_steps == 0:
                    # Evaluation 
                    model.eval()
                    true_max_positions = []
                    true_min_positions = []

                    pred_max_positions = []
                    pred_min_positions = []
                    for batch in tqdm(eval_dataloader, desc="Evaluating..", leave=False):
                        with torch.no_grad():
                            outputs = model(input_ids=batch["input_ids"].to(device), 
                                attention_mask=batch["attention_mask"].to(device))
                            max_positions = np.argmax(outputs.max_logits.cpu().numpy(), axis=-1)
                            min_positions = np.argmax(outputs.min_logits.cpu().numpy(), axis=-1)

                        pred_max_positions.extend(max_positions.tolist())
                        pred_min_positions.extend(min_positions.tolist())

                        max_labels = batch["max_csep_index"].squeeze().cpu().numpy().tolist()
                        min_labels = batch["min_csep_index"].squeeze().cpu().numpy().tolist()

                        true_max_positions.extend(max_labels)
                        true_min_positions.extend(min_labels)
                    
                    # Calculate the accuracy of prediction
                    max_correct = 0
                    min_correct = 0
                    both_correct = 0
                    for max_pred, min_pred, max_true, min_true in zip(pred_max_positions, pred_min_positions, true_max_positions, true_min_positions):
                        if (max_pred == max_true):
                            max_correct += 1
                        if (min_pred == min_true):
                            min_correct += 1
                        if (max_pred == max_true) and (min_pred == min_true):
                            both_correct += 1

                    max_accuracy = max_correct / len(true_max_positions)
                    min_accuracy = min_correct / len(true_min_positions)
                    both_accuracy = both_correct / len(true_max_positions)

                    metrics = {
                        "max_accuracy": max_accuracy,
                        "min_accuracy": min_accuracy,
                        "both_accuracy": both_accuracy
                    }
                    wandb.log(metrics)
                    if both_accuracy > best_accuracy:
                        # Save the model to output dir
                        logger.info("Best checkpoint found at step {}".format(global_step))
                        best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
                        if not os.path.exists(best_checkpoint_dir):
                            os.makedirs(best_checkpoint_dir)
                        logger.info("Saving model checkpoint to %s", best_checkpoint_dir)
                        model.module.save_pretrained(best_checkpoint_dir)
                        tokenizer.save_pretrained(best_checkpoint_dir)
                        torch.save(args, os.path.join(best_checkpoint_dir, "training_args.bin"))
                        best_accuracy = both_accuracy

                if global_step >= args.num_train_steps:
                    logger.info("Training finished. Total steps: {}".format(global_step))
                    break 
                model.train()
            elif args.objective == "minmaxmedian":
                outputs = model(input_ids=batch["input_ids"].to(device), 
                                attention_mask=batch["attention_mask"].to(device), 
                                max_csep_index=batch["max_csep_index"].to(device), 
                                min_csep_index=batch["min_csep_index"].to(device),
                                median_csep_index=batch["median_csep_index"].to(device)
                            )
                loss = outputs.loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step +=1 
                
                if global_step % args.logging_steps == 0:
                    # Logging 
                    wandb.log({"loss":loss})

                if global_step % args.evaluation_steps == 0:
                    # Evaluation 
                    model.eval()
                    true_max_positions = []
                    true_min_positions = []
                    true_median_positions = []

                    pred_max_positions = []
                    pred_min_positions = []
                    pred_median_positions = []
                    for batch in tqdm(eval_dataloader, desc="Evaluating..", leave=False):
                        with torch.no_grad():
                            outputs = model(input_ids=batch["input_ids"].to(device), 
                                attention_mask=batch["attention_mask"].to(device))
                            max_positions = np.argmax(outputs.max_logits.cpu().numpy(), axis=-1)
                            min_positions = np.argmax(outputs.min_logits.cpu().numpy(), axis=-1)
                            median_positions = np.argmax(outputs.median_logits.cpu().numpy(), axis=-1)

                        pred_max_positions.extend(max_positions.tolist())
                        pred_min_positions.extend(min_positions.tolist())
                        pred_median_positions.extend(median_positions.tolist())

                        max_labels = batch["max_csep_index"].squeeze().cpu().numpy().tolist()
                        min_labels = batch["min_csep_index"].squeeze().cpu().numpy().tolist()
                        median_labels = batch["median_csep_index"].squeeze().cpu().numpy().tolist()

                        true_max_positions.extend(max_labels)
                        true_min_positions.extend(min_labels)
                        true_median_positions.extend(median_labels)
                    
                    # Calculate the accuracy of prediction
                    max_correct = 0
                    min_correct = 0
                    median_correct = 0
                    all_correct = 0
                    for max_pred, min_pred, median_pred, max_true, min_true, median_true in zip(pred_max_positions, pred_min_positions, pred_median_positions, true_max_positions, true_min_positions, true_median_positions):
                        if (max_pred == max_true):
                            max_correct += 1
                        if (min_pred == min_true):
                            min_correct += 1
                        if (median_pred == median_true):
                            median_correct += 1
                        if (max_pred == max_true) and (min_pred == min_true) and (median_pred == median_true):
                            all_correct += 1

                    max_accuracy = max_correct / len(true_max_positions)
                    min_accuracy = min_correct / len(true_min_positions)
                    median_accuracy = median_correct / len(true_median_positions)
                    all_accuracy = all_correct / len(true_max_positions)

                    metrics = {
                        "max_accuracy": max_accuracy,
                        "min_accuracy": min_accuracy,
                        "median_accuracy": median_accuracy,
                        "all_accuracy": all_accuracy
                    }
                    wandb.log(metrics)
                    if all_accuracy > best_accuracy:
                        # Save the model to output dir
                        logger.info("Best checkpoint found at step {}".format(global_step))
                        best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
                        logger.info("Saving model checkpoint to %s", best_checkpoint_dir)
                        model.module.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        best_accuracy = all_accuracy

                if global_step >= args.num_train_steps:
                    logger.info("Training finished. Total steps: {}".format(global_step))
                    break 
                model.train()
            elif args.objective == "median":
                outputs = model(input_ids=batch["input_ids"].to(device), 
                                attention_mask=batch["attention_mask"].to(device), 
                                median_csep_index=batch["median_csep_index"].to(device)
                            )
                loss = outputs.loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step +=1 
                
                if global_step % args.logging_steps == 0:
                    # Logging 
                    wandb.log({"loss":loss})

                if global_step % args.evaluation_steps == 0:
                    # Evaluation 
                    model.eval()
                    true_median_positions = []

                    pred_median_positions = []
                    for batch in tqdm(eval_dataloader, desc="Evaluating..", leave=False):
                        with torch.no_grad():
                            outputs = model(input_ids=batch["input_ids"].to(device), 
                                attention_mask=batch["attention_mask"].to(device))
                            median_positions = np.argmax(outputs.median_logits.cpu().numpy(), axis=-1)

                        pred_median_positions.extend(median_positions.tolist())
                        median_labels = batch["median_csep_index"].squeeze().cpu().numpy().tolist()
                        true_median_positions.extend(median_labels)
                    
                    # Calculate the accuracy of prediction
                    median_correct = 0
                    all_correct = 0
                    for median_pred, median_true in zip(pred_median_positions, true_median_positions):
                        if (median_pred == median_true):
                            median_correct += 1


                    median_accuracy = median_correct / len(true_median_positions)

                    metrics = {
                        "median_accuracy": median_accuracy,
                    }
                    all_accuracy = median_accuracy 
                    wandb.log(metrics)
                    if all_accuracy > best_accuracy:
                        # Save the model to output dir
                        logger.info("Best checkpoint found at step {}".format(global_step))
                        best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
                        logger.info("Saving model checkpoint to %s", best_checkpoint_dir)
                        model.module.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        best_accuracy = all_accuracy

                if global_step >= args.num_train_steps:
                    logger.info("Training finished. Total steps: {}".format(global_step))
                    break 
                model.train()

            else:
                raise NotImplementedError

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="roberta-base", type=str)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--objective", default="minmax", choices=["minmax", "minmaxmedian", "median"])

    parser.add_argument("--wikitable_path", default="../data_wikitable/all_plain_tables.json", type=str)
    parser.add_argument("--output_dir", help="Path to save the trained checkpoints", default="../models/trained_models/RATE")

    # Training arguments
    parser.add_argument("--num_train_steps", default=10000, type=int)
    parser.add_argument("--evaluation_steps", default=500, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--evaluation_batch_size", default=128, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)

    # Other arguments
    parser.add_argument("--logging_steps", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(args)