# coding: UTF-8
import os
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn import metrics

from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from tools.common import seed_everything
from tools.common import logger
from callback.progressbar import ProgressBar
from metrics.glue_compute_metrics import compute_metrics

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'


def build_vocab(examples, tokenizer, max_size, min_freq):
    vocab_dic = {}
    for example in examples:
        for word in tokenizer(example.text_a):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        vocab_size, features = torch.load(cached_features_file)
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

        vocab_path = os.path.join(args.model_name_or_path, 'vocab.pkl')
        if os.path.exists(vocab_path):
            vocab = pkl.load(open(vocab_path, 'rb'))
        else:
            train_example = args.processor.get_train_examples(args.data_dir) if data_type != 'train' else examples
            vocab = build_vocab(train_example, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
            pkl.dump(vocab, open(vocab_path, 'wb'))
        vocab_size = len(vocab)
        logger.info(f"Vocab size: {vocab_size}")

        features = []
        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d" % ex_index)
            content, num, label = example.text_a, example.num, example.label
            words_line = []
            token = tokenizer(content)
            seq_len = len(token)
            if len(token) < args.max_seq_length:
                token.extend([PAD] * (args.max_seq_length - len(token)))
            else:
                token = token[:args.max_seq_length]
                seq_len = args.max_seq_length
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))

            if args.model_type == 'FastText':
                # fasttext ngram
                buckets = args.n_gram_vocab
                bigram = []
                trigram = []
                # ------ngram------
                for i in range(args.max_seq_length):
                    bigram.append(biGramHash(words_line, i, buckets))
                    trigram.append(triGramHash(words_line, i, buckets))
                # -----------------
                features.append((words_line, bigram, trigram, seq_len, int(label), num))
            else:
                features.append((words_line, 0, 0, seq_len, int(label), num))

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save((vocab_size, features), cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_bigram = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_trigram = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)
    if args.output_mode == "classification":
        all_labels = torch.tensor([f[4] for f in features], dtype=torch.long)
    elif args.output_mode == "regression":
        all_labels = torch.tensor([f[4] for f in features], dtype=torch.float)
    all_nums = torch.tensor([f[5] for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_bigram, all_trigram, all_lens, all_labels, all_nums)
    return vocab_size, dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)

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

    # Train!
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
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training {}|{}'.format(_ + 1, int(args.num_train_epochs)))
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type == 'FastText':
                inputs = {'input_ids': batch[0],
                          'bigram': batch[1],
                          'trigram': batch[2],
                          'num': batch[5],
                          'labels': batch[4]}
            else:
                inputs = {'input_ids': batch[0],
                          'num': batch[5],
                          'labels': batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            pbar(step, {'loss': loss.item()})

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                evaluate(args, model, tokenizer)

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


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        _, eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, data_type='dev')
        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        scores = None
        pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                if args.model_type == 'FastText':
                    inputs = {'input_ids': batch[0],
                              'bigram': batch[1],
                              'trigram': batch[2],
                              'labels': batch[4],
                              'num': batch[5]}
                else:
                    inputs = {'input_ids': batch[0],
                              'labels': batch[4],
                              'num': batch[5]}
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

        # report = metrics.classification_report(out_label_ids, preds, target_names=args.class_list, digits=4)
        # precision, recall, thresholds = metrics.precision_recall_curve(out_label_ids, preds)
    return results
