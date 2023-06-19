# -*- coding: utf-8 -*-
import argparse
import os.path

import pandas as pd
import json
import copy
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import onnxruntime as ort

from models.albert import Model, Config


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, num=None, time_line=None, send_role=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.num = num
        self.time_line = time_line
        self.send_role = send_role
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len, position_ids=None, num=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.input_len = input_len
        self.label = label
        self.num = num

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _truncate_seq_pair(tokens_tol, tokens_usr, tokens_svr, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        lens = [len(tokens_tol), len(tokens_usr), len(tokens_svr)]
        total_length = sum(lens)
        if total_length <= max_length:
            break
        if lens.index(max(lens)) == 0:
            tokens_tol.pop(0)
        elif lens.index(max(lens)) == 1:
            tokens_usr.pop(0)
        else:
            tokens_svr.pop(0)


def convert_examples_to_features(examples, tokenizer, max_seq_length=512, label_list=None):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_seq_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    if label_list is not None:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    all_lens = []
    for (ex_index, example) in enumerate(examples):
        try:
            usr_idx = example.text_a.index('[USR]')
            svr_idx = example.text_a.index('[SVR]')
        except:
            continue
        tokens_tol, positions_tol = [], []
        positions_tmp = [max(min(int(t), 511), 0) for t in example.time_line.split('[SEP]')]
        send_role_tmp = [int(t) for t in example.send_role.split('[SEP]')]
        tokens_tmp = example.text_a[:usr_idx].split('[SEP]')
        assert len(positions_tmp) == len(tokens_tmp)
        assert len(send_role_tmp) == len(tokens_tmp)
        for i, (pos, line, sr) in enumerate(zip(positions_tmp, tokens_tmp, send_role_tmp)):
            tmp = tokenizer.tokenize(line) + ['[SEP]']
            tokens_tol.extend(tmp)
            if i > 0: positions_tol[-1] = sr
            positions_tol.extend([pos] * len(tmp))
        tokens_usr = [] if usr_idx + 5 == svr_idx else tokenizer.tokenize(example.text_a[usr_idx+5:svr_idx])
        tokens_svr = [] if svr_idx + 5 == len(example.text_a) else tokenizer.tokenize(example.text_a[svr_idx+5:])
        all_lens.append(len(tokens_tol) + len(tokens_usr) + len(tokens_svr))

        # Account for [CLS], [SEP], [USR], [SEP], [SVR], [SEP] with "- 6"
        _truncate_seq_pair(tokens_tol, tokens_usr, tokens_svr, max_seq_length - 6)
        positions_tol = positions_tol[-len(tokens_tol):]

        tokens = []
        token_type_ids = []
        position_ids = []
        tokens.append("[CLS]")
        token_type_ids.append(0)
        position_ids.append(0)
        for pid, token in zip(positions_tol, tokens_tol):
            tokens.append(token)
            token_type_ids.append(0)
            position_ids.append(pid)
        tokens.append("[SEP]")
        token_type_ids.append(0)
        position_ids.append(positions_tol[-1])

        tokens.append("[USR]")
        token_type_ids.append(1)
        position_ids.append(0)
        for token in tokens_usr:
            tokens.append(token)
            token_type_ids.append(1)
            position_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(1)
        position_ids.append(0)

        tokens.append("[SVR]")
        token_type_ids.append(2)
        position_ids.append(0)
        for token in tokens_svr:
            tokens.append(token)
            token_type_ids.append(2)
            position_ids.append(0)
        tokens.append("[SEP]")
        token_type_ids.append(2)
        position_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        input_len = len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
            position_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(position_ids) == max_seq_length

        label_id = label_map[example.label] if label_list is not None else None

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          label=label_id,
                          input_len=input_len,
                          num=example.num))
    return features


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    # all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_nums, all_position_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_position_ids = all_position_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_nums, all_position_ids


def compute_metrics(labels, scores):
    from sklearn import metrics
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    pre_thresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    idx = [np.argmin(np.abs(precision[:-1] - p)) for p in pre_thresh]
    df = pd.DataFrame({'precision': precision[idx], 'recall': recall[idx], 'thresholds': thresholds[idx]})
    df = pd.DataFrame(df.values.T, index=df.columns, columns=pre_thresh)
    return {
        'acc': metrics.accuracy_score(labels, scores > 0.5),
        'auc': metrics.roc_auc_score(labels, scores > 0.5),
        'f1-score': metrics.f1_score(labels, scores > 0.5),
        'report': '\n{}'.format(metrics.classification_report(labels, scores > 0.5)),
        'confusion': '\n{}'.format(metrics.confusion_matrix(labels, scores > 0.5)),
        'pr-curve': '\n{}'.format(df),
    }


def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    args = parser.parse_args()
    return args


def load_dataset(data_path, length=100):
    df = pd.read_csv(data_path, usecols=['session_id', 'last_message_time',
        'content', 'polling', 'user_message_time', 'user_wait_time', 'user_message_cnt', 'user_recovery_time',
        'servicer_message_time', 'servicer_wait_time', 'servicer_message_cnt', 'servicer_recovery_time',
        'send_role', 'time_line', 'send_role_list', 'label'], nrows=length)
    sids = df['session_id'].values
    lst_time = df['last_message_time'].values
    data = df.values[:, 2:]
    examples = []
    for i in range(df.shape[0]):
        num = [float(n) for n in data[i][1:-3]]
        # label = str(data[i][-1])
        label = '1' if lst_time[i] <= data[i][1] else str(data[i][-1])
        examples.append(InputExample(guid=sids[i], text_a=data[i][0], text_b=None, num=num, time_line=data[i][-3],
                                     send_role=data[i][-2], label=label))
    return examples


def build_model_inputs(data, config, args):
    features = convert_examples_to_features(data, config.tokenizer, max_seq_length=args.max_seq_length,
                                            label_list=args.label_list)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_position_ids = torch.tensor([f.position_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_nums = torch.tensor([f.num for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels, all_nums,
                            all_position_ids)
    return dataset


def export_onnx_model(model, args):
    # if os.path.exists(args.onnx_path):
    #     return ort.InferenceSession(args.onnx_path)

    dummy_input = {
        'input_ids': torch.randint(0, 1, (args.batch_size, args.max_seq_length)),
        'attention_mask': torch.randint(0, 1, (args.batch_size, args.max_seq_length)),
        'token_type_ids': torch.randint(0, 1, (args.batch_size, args.max_seq_length)),
        'position_ids': torch.randint(0, 1, (args.batch_size, args.max_seq_length)),
        'num': torch.rand((args.batch_size, 10))
    }
    dummy_axes = {
        'input_ids': {0: 'batch_size', 1: 'seq_length'},  # 0, 1分别代表axis 0和axis 1
        'attention_mask': {0: 'batch_size', 1: 'seq_length'},
        'token_type_ids': {0: 'batch_size', 1: 'seq_length'},
        'position_ids': {0: 'batch_size', 1: 'seq_length'},
        'num': {0: 'batch_size', 1: 'seq_length'},
        'outputs': {0: 'batch_size', 1: 'seq_length'}
    }  # 用于变长序列（比如dynamic padding）和可能改变batch size的情况
    output_names = ['outputs']

    torch.onnx.export(
        model,
        args=tuple(dummy_input.values()),
        f=args.onnx_path,
        opset_version=11,  # 此处有坑，必须指定≥10，否则会报错
        do_constant_folding=True,
        input_names=list(dummy_input),
        output_names=output_names,
        export_params=True,
        dynamic_axes=dummy_axes
    )
    return ort.InferenceSession(args.onnx_path)


def export_tf_model(args):
    if os.path.exists(args.tf_model_path):
        return
    from onnx_tf.backend import prepare
    import onnx

    onnx_model = onnx.load(args.onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(args.tf_model_path)


def export_tf_lite_model(args):
    if os.path.exists(args.tf_lite_path):
        return
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(args.tf_model_path,
                                                         input_arrays=['input_ids', 'attention_mask', 'token_type_ids', 'position_ids', 'num'],
                                                         input_shapes={'input_ids': [1, 256], 'attention_mask': [1, 256], 'token_type_ids': [1, 256], 'position_ids': [1, 256], 'num': [1, 256]},
                                                         output_arrays=['outputs'])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(args.tf_lite_path, 'wb') as f:
        f.write(tf_lite_model)


def predict(model, dataloader, args, mode='pytorch'):
    nb_eval_steps = 0
    out_label_ids, scores = None, None
    cost_time = time.time()
    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'num': batch[4]}
            inputs['token_type_ids'] = batch[2]
            inputs['position_ids'] = batch[5]
            if mode == 'pytorch':
                outputs = model(**inputs)
                logits = outputs[0]
            elif mode == 'onnx':
                outputs = model.run(output_names=['outputs'],
                                    input_feed={key: value.numpy() for key, value in inputs.items()})[0]
                logits = torch.from_numpy(outputs[:, :2])

        nb_eval_steps += 1
        if scores is None:
            out_label_ids = batch[3].detach().cpu().numpy()
            scores = torch.nn.Softmax(dim=1)(logits.detach().cpu()).numpy()
        else:
            out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)
            scores = np.append(scores, torch.nn.Softmax(dim=1)(logits.detach().cpu()).numpy(), axis=0)
    scores = scores[:, 1]
    result = compute_metrics(out_label_ids, scores)

    cost_time = (time.time() - cost_time) * 1000
    print('\ntotal {} samples. cost {:.2f} ms. batch time {:.2f} ms. avg sample time {:.2f} ms'.format(
        args.batch_size * len(dataloader), cost_time, cost_time / len(dataloader), cost_time / args.batch_size / len(dataloader)))
    for key in sorted(result.keys()):
        print("{} = {}".format(key, str(result[key])))
    return scores


def save_predict_result(data, scores):
    records = []
    for d, s in zip(data, scores):
        res = 'right' if (d.label == '1' and s > 0.5) or (d.label == '0' and s <= 0.5) else 'error'
        records.append([d.guid, d.text_a, d.label, s, res])
    df = pd.DataFrame(records, columns=['sid', 'text', 'label', 'score', 'result'])
    df.to_csv('preds_eval.csv', index=False)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name_or_path = 'prev_trained_model/albert_base_zh'
    args.max_seq_length = 256
    args.batch_size = 1

    task_name = 'smart_boost'
    data_path = 'dataset/smart_boost/dev.csv'
    # args.model_path = 'outputs/smart_boost_output/albert_base/checkpoint-375000'
    # args.model_path = 'outputs/smart_boost_output/albert_small/checkpoint-375000'
    args.model_path = 'outputs/smart_boost_output/albert_tiny/checkpoint-187500'

    args.onnx_path = os.path.join(args.model_path, 'inference_model.onnx')
    args.tf_model_path = os.path.join(args.model_path, 'inference_tf_model')
    args.tf_lite_path = os.path.join(args.model_path, 'inference_model.tflite')

    # Prepare GLUE task
    config = Config(args, finetuning_task=task_name)
    args.class_list = config.class_list
    args.label_list = [str(_) for _ in range(config.num_labels)]

    data_src = load_dataset(data_path, length=1000)
    data_intputs = build_model_inputs(data_src, config, args)
    config.n_num_feat = len(data_intputs[0][5])

    sampler = SequentialSampler(data_intputs)
    dataloader = DataLoader(data_intputs, sampler=sampler, batch_size=args.batch_size, collate_fn=collate_fn)

    model_torch = Model.from_pretrained(args.model_path).eval()
    # scores_torch = predict(model_torch, dataloader, args, mode='pytorch')

    model_onnx = export_onnx_model(model_torch, args)
    scores_onnx = predict(model_onnx, dataloader, args, mode='onnx')
    # np.testing.assert_almost_equal(scores_torch, scores_onnx, decimal=5)

    save_predict_result(data_src, scores_onnx)

    # export_tf_model(args)
    # export_tf_lite_model(args)
