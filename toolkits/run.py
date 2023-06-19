# coding: UTF-8
import os
import time
import glob
import torch
import numpy as np
from importlib import import_module
import argparse
from tools.common import seed_everything, init_logger, logger
from processors import glue_output_modes as output_modes
from processors import glue_processors as processors
from pytorch_pretrained.file_utils import WEIGHTS_NAME


def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument('--model_type', type=str, required=True,
                        help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer, bert, ERNIE, albert')
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--task_name", default='THUCNews', type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--embedding_type', default='random', type=str,
                        help='random or pre_trained')
    parser.add_argument('--use_word', action='store_true',
                        help='True for word, False for char')

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument('--logging_steps', type=int, default=None,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=None,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_checkpoints", type=int, default=None,
                        help="Evaluate specific checkpoints")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args


def get_model_type(model_type):
    model_type = model_type.lower()
    if 'bert' in model_type or 'ernie' == model_type:
        return 'bert'
    if 'transformer' in model_type:
        return 'transformer'
    else:
        return 'else'


if __name__ == '__main__':
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.output_dir = os.path.join(args.output_dir, '{}'.format(args.model_type))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    init_logger(log_file=os.path.join(args.output_dir, '{}-{}.log'.format(args.model_type, args.task_name)))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Prepare GLUE task
    # args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]

    module = import_module('models.' + args.model_type)
    config = module.Config(args, finetuning_task=args.task_name)
    args.class_list = config.class_list

    if get_model_type(args.model_type) == 'bert':
        from train_eval_bert import load_and_cache_examples, train, evaluate
    else:
        from train_eval import load_and_cache_examples, train, evaluate
        if args.model_type == 'FastText':
            args.n_gram_vocab = config.n_gram_vocab

    # Training
    if args.do_train:
        if get_model_type(args.model_type) == 'bert':
            train_dataset = load_and_cache_examples(args, args.task_name, config.tokenizer, data_type='train')
            if train_dataset[0][5] is not None:
                config.n_num_feat = len(train_dataset[0][5])
        else:
            config.n_vocab, train_dataset = load_and_cache_examples(args, args.task_name, config.tokenizer, data_type='train')
            if train_dataset[0][5] is not None:
                config.n_num_feat = len(train_dataset[0][5])

        # model = module.Model(config).from_pretrained(args.model_name_or_path, config=config).to(args.device)
        model = module.Model.from_pretrained(args.model_name_or_path, config=config).to(args.device)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, config.tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_step = %s", global_step, tr_loss, best_steps)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    # Evaluation
    results = []
    if args.do_eval or args.do_predict:
        if args.eval_checkpoints is not None:
            checkpoints = [(args.eval_checkpoints, os.path.join(args.output_dir, 'checkpoint-{}'.format(args.eval_checkpoints)))]
        elif args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [(int(checkpoint.split('-')[-1]), checkpoint)
                           for checkpoint in checkpoints if checkpoint.find('checkpoint') != -1]
            checkpoints = sorted(checkpoints, key=lambda x: x[0])
        else:
            checkpoints = [(0, args.output_dir)]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for _, checkpoint in checkpoints:
            global_step = checkpoint.split('/')[-1].split('-')[-1] if len(checkpoints) > 0 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            # model = module.Model(config).from_pretrained(checkpoint)
            model = module.Model.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, config.tokenizer, global_step, prefix=prefix, data_type='dev' if args.do_eval else 'test')
            results.extend([(k + '_{}'.format(global_step), v) for k, v in result.items()])
        output_eval_file = os.path.join(args.output_dir, "checkpoint_eval_results_{}.txt".format('dev' if args.do_eval else 'test'))
        with open(output_eval_file, "w") as writer:
            for key, value in results:
                writer.write("%s = %s\n" % (key, str(value)))
