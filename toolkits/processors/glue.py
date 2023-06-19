""" GLUE processors and helpers """

import logging
import os
import torch
from .utils import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


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


def glue_convert_examples_to_features(examples, tokenizer,
                                      max_seq_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_seq_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    all_lens = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            # logger.info("Writing example %d" % (ex_index))
            max_len = 0
            if len(all_lens) > 0:
                all_lens.sort()
                max_len = all_lens[int(len(all_lens) * 0.95)]
                logger.info("Writing example %d. Input ids 95%% len %.2f. Mean len %.2f. Sample number %d" % (
                    ex_index, max_len, sum(all_lens)/len(all_lens), len(all_lens)))

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
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("position_ids: %s" % " ".join([str(x) for x in position_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("input length: %d" % (input_len))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          position_ids=position_ids,
                          label=label_id,
                          input_len=input_len,
                          num=example.num))
    return features


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


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the cola data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the sts-b data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class LcqmcProcessor(DataProcessor):
    """Processor for the LCQMC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            if set_type == 'test222':
                label = '0'
            else:
                label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class THUCNewsProcessor(DataProcessor):
    """Processor for the SmartBoost data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.txt"), delimiter='\t'), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.txt"), delimiter='\t'), "dev")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(10)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SmartBoostProcessor(DataProcessor):
    """Processor for the SmartBoost data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv"), delimiter=','), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv"), delimiter=','), "dev")

    def get_labels(self):
        """See base class."""
        # return ["active", "inactive"]
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SmartBoostNumProcessor(DataProcessor):
    """Processor for the SmartBoostNumProcessor data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "train.csv"), delimiter=','), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "dev.csv"), delimiter=','), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "test.csv"), delimiter=','), "test")

    def get_labels(self):
        """See base class."""
        # return ["active", "inactive"]
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            mark = line[:2]
            line = line[2:]
            text_a = line[0]
            label = line[-1]
            time_line = line[-3]
            send_role_list = line[-2]
            num = [float(n) for n in line[1:-3]]
            idx = text_a.index('[USR]')
            assert len(time_line.split('[SEP]')) == len(text_a[:idx].split('[SEP]'))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, num=num, time_line=time_line, send_role=send_role_list, label=label))
        return examples


glue_tasks_num_labels = {
    'mnli': 3,
    'mrpc': 2,
    'sst-2': 2,
    'sts-b': 1,
    'qqp': 2,
    'qnli': 2,
    'rte': 2,
    'lcqmc': 2,
    'xnli': 3,
}

glue_processors = {
    'cola': ColaProcessor,
    'mnli': MnliProcessor,
    'mnli-mm': MnliMismatchedProcessor,
    'mrpc': MrpcProcessor,
    'sst-2': Sst2Processor,
    'sts-b': StsbProcessor,
    'qqp': QqpProcessor,
    'qnli': QnliProcessor,
    'rte': RteProcessor,
    'lcqmc': LcqmcProcessor,
    'wnli': WnliProcessor,
    'THUCNews': THUCNewsProcessor,
    # 'smart_boost': SmartBoostProcessor,
    'smart_boost': SmartBoostNumProcessor,
}

glue_output_modes = {
    'cola': 'classification',
    'mnli': 'classification',
    'mnli-mm': 'classification',
    'mrpc': 'classification',
    'sst-2': 'classification',
    'sts-b': 'regression',
    'qqp': 'classification',
    'qnli': 'classification',
    'rte': 'classification',
    'wnli': 'classification',
    'lcqmc': 'classification',
    'THUCNews': 'classification',
    'smart_boost': 'classification',
}
