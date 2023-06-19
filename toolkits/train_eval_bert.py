import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import time

from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from metrics.glue_compute_metrics import compute_metrics
from processors import glue_convert_examples_to_features as convert_examples_to_features
from processors import collate_fn
from tools.common import seed_everything
from tools.common import logger
from callback.progressbar import ProgressBar


def plot_pred_dist(labels, preds, save_path=None):
    path, name = os.path.split(save_path)

    # distribution
    x = pd.DataFrame({"labels": labels, "preds": preds})
    class_zero_preds = x.loc[x["labels"] == 0, "preds"]
    class_one_preds = x.loc[x["labels"] == 1, "preds"]

    plt.figure()
    plt.hist(x['preds'], bins=20, label='all', alpha=0.2)
    plt.hist(class_zero_preds, bins=20, label="0", alpha=0.5)
    plt.hist(class_one_preds, bins=20, label="1", alpha=0.5)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(path, 'dist_{}'.format(name)))

    # pr-curve
    precision, recall, _ = metrics.precision_recall_curve(labels, preds)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    if save_path:
        plt.savefig(os.path.join(path, 'pr_curve_{}'.format(name)))
    plt.close()


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = args.processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and 'roberta' in args.model_type:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        if data_type == 'train':
            examples = args.processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = args.processor.get_dev_examples(args.data_dir)
        else:
            examples = args.processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.max_seq_length,
                                                output_mode=args.output_mode)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if args.output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    all_nums = torch.tensor([f.num for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_nums, all_position_ids)
    return dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,
                                  collate_fn=collate_fn)

    if args.logging_steps is None:
        args.logging_steps = len(train_dataloader)
    if args.save_steps is None:
        args.save_steps = len(train_dataloader)

    num_training_steps = len(train_dataloader) * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_training_steps)

    # Trainoutput_dir!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    dev_best_loss = float('inf')
    dev_best_steps = None
    model.zero_grad()
    seed_everything(args.seed)
    for _ in range(int(args.num_train_epochs)):
        preds, out_label_ids, scores, records = None, None, None, None
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training {}|{}'.format(_ + 1, int(args.num_train_epochs)))
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'num': batch[4],
                      'labels': batch[3]}
            inputs['token_type_ids'] = batch[2]
            inputs['position_ids'] = batch[5]
            outputs = model(**inputs)
            # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss, logits = outputs[:2]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            # evaluate train dataset
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                scores = torch.nn.Softmax(dim=1)(logits.detach().cpu()).numpy()
                records = inputs['num'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                scores = np.append(scores, torch.nn.Softmax(dim=1)(logits.detach().cpu()).numpy(), axis=0)
                records = np.append(records, inputs['num'].detach().cpu().numpy(), axis=0)

            global_step += 1
            pbar(step, {'loss': loss.item()})

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info("\n***** Running train evaluation {} *****".format(global_step))
                if args.output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                    scores = scores[:, 1]
                elif args.output_mode == "regression":
                    preds = np.squeeze(preds)
                result = compute_metrics(args.task_name, preds, out_label_ids, scores)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                plot_pred_dist(out_label_ids, scores, save_path=os.path.join(output_dir, 'eval_train.png'))
                # save prediction result
                df = pd.DataFrame(records,
                                  columns=['polling', 'user_message_time', 'user_wait_time', 'user_message_cnt',
                                           'user_recovery_time', 'servicer_message_time', 'servicer_wait_time',
                                           'servicer_message_cnt', 'servicer_recovery_time', 'send_role'])
                df['label'] = out_label_ids
                df['preds'] = scores
                df.to_csv(os.path.join(output_dir, 'preds_train.csv'), index=False)
                preds, out_label_ids, scores = None, None, None
                # Log metrics
                evaluate(args, model, tokenizer, global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if loss.item() < dev_best_loss:
                    dev_best_loss = loss.item()
                    dev_best_steps = global_step
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step, dev_best_steps


def evaluate(args, model, tokenizer, global_step, prefix="", data_type='dev'):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type=data_type)
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size,
                                     collate_fn=collate_fn)

        # Eval!
        logger.info("***** Running {} evaluation {} *****".format(data_type, prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds, out_label_ids, scores = None, None, None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating {}".format(global_step))
        cost_time = time.time()
        for step, batch in enumerate(eval_dataloader):
            # test predict time
            if len(batch[0])*step >= 100:
                cost_time = (time.time() - cost_time) * 1000
                print('\ntotal {} samples. cost {:.2f} ms. batch time {:.2f} ms. avg sample time {:.2f} ms'.format(
                    len(batch[0])*step, cost_time,cost_time/step, cost_time/len(batch[0])/step))
                return {}
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            # print(batch[0].shape, batch[1].shape, batch[2].shape, batch[4].shape)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'num': batch[4],
                          'labels': batch[3]}
                inputs['token_type_ids'] = batch[2]
                inputs['position_ids'] = batch[5]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                scores = torch.nn.Softmax(dim=1)(logits.detach().cpu()).numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                scores = np.append(scores, torch.nn.Softmax(dim=1)(logits.detach().cpu()).numpy(), axis=0)
            pbar(step)
        print(' ')
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            scores = scores[:, 1]
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids, scores)
        results.update(result)
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot_pred_dist(out_label_ids, scores, save_path=os.path.join(output_dir, 'eval_{}.png'.format(data_type)))

        # save prediction result
        df = pd.DataFrame(eval_dataset[:][5].numpy(),
                          columns=['polling', 'user_message_time', 'user_wait_time', 'user_message_cnt',
                                   'user_recovery_time', 'servicer_message_time', 'servicer_wait_time',
                                   'servicer_message_cnt', 'servicer_recovery_time', 'send_role'])
        df['label'] = out_label_ids
        df['preds'] = scores
        df.to_csv(os.path.join(output_dir, 'preds_{}.csv'.format(data_type)), index=False)
    return results
