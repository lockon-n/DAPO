# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Dialog_engagement processors and helpers """

import logging
import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
import tqdm

from file_utils import is_tf_available
from tokenization_utils import PreTrainedTokenizer
from processer_utils import DataProcessor, InputExample, InputFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def dialog_evaluation_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    is_nsp=False,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: DialogEvaluation task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _dialog_evaluation_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode,
        is_nsp=is_nsp
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.

        """
        processor = dialog_evaluation_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = dialog_evaluation_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                d = {k: v for k, v in asdict(ex).items() if v is not None}
                label = d.pop("label")
                yield (d, label)

        input_names = ["input_ids"] + tokenizer.model_input_names

        return tf.data.Dataset.from_generator(
            gen,
            ({k: tf.int32 for k in input_names}, tf.int64),
            ({k: tf.TensorShape([None]) for k in input_names}, tf.TensorShape([])),
        )


def _dialog_evaluation_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
    is_nsp=False,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = dialog_evaluation_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = dialog_evaluation_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    if is_nsp:
        ts='longest_first'
    else:
        ts='only_first_from_head'
    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation_strategy=ts,
    )

    features = []
    for i in tqdm.tqdm(range(len(examples))):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"

'''
class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = None if set_type.startswith("test") else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")), "test_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        if test_mode:
            lines = lines[1:]
        text_index = 1 if test_mode else 3
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 1 if set_type == "test" else 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
        q1_index = 1 if test_mode else 3
        q2_index = 2 if test_mode else 4
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[q1_index]
                text_b = line[q2_index]
                label = None if test_mode else line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
'''

class Convai2EngagementProcessor(DataProcessor):
    """Processor for the Convai2Engagement data set ."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["history"].numpy().decode("utf-8"),
            tensor_dict["response"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,5]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "test" else line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RawtextProcessor(DataProcessor):
    """Processor for the Rawtext dialog for EvaluationPretraining ."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line)!=7:
                continue
            guid = "%s-%s" % (set_type, i)
            inst = line[0]
            text_a = line[1]
            if "U" in inst:
                label_ = 0.0
            else:
                label_ = (float(line[2])+float(line[3])+float(line[4]))/3
            label = None if set_type == "test" else label_
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class RawtextWithoutNIDFProcessor(DataProcessor):
    """Processor for the Rawtext dialog for EvaluationPretraining ."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 7:
                continue
            guid = "%s-%s" % (set_type, i)
            inst=line[0]
            text_a = line[1]
            if "U" in inst:
                label_ = 0.0
            else:
                label_ = 1.0
            label = None if set_type == "test" else label_
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Rawtext_1_Processor(DataProcessor):
    """Processor for the Rawtext dialog for EvaluationPretraining ."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line)!=7:
                continue
            guid = "%s-%s" % (set_type, i)
            inst=line[0]
            text_a = line[1]
            if "U" in inst:
                label_=0.0
            else:
                label_=float(line[2])
            label = None if set_type == "test" else label_
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Rawtext_2_Processor(DataProcessor):
    """Processor for the Rawtext dialog for EvaluationPretraining ."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) != 7:
                continue
            guid = "%s-%s" % (set_type, i)
            inst = line[0]
            text_a = line[1]
            if "U" in inst:
                label_ = 0.0
            else:
                label_ = float(line[3])
            label = None if set_type == "test" else label_
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class Rawtext_3_Processor(DataProcessor):
    """Processor for the Rawtext dialog for EvaluationPretraining ."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line)!=7:
                continue
            guid = "%s-%s" % (set_type, i)
            inst=line[0]
            text_a = line[1]
            if "U" in inst:
                label_=0.0
            else:
                label_=float(line[4])
            label = None if set_type == "test" else label_
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class MutualProcessor(DataProcessor):
    """Processor for the Mutual Dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, e) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            answer = e['answers']
            if answer == "A": answerid = 0
            elif answer == "B": answerid = 1
            elif answer == "C": answerid = 2
            else: answerid = 3
            history = e['article']
            for j in range(4):
                dialog=history+e["options"][j]
                guid_=guid+"-"+str(j)
                if j==answerid:label_=1
                else: label_= 0
                label = None if set_type == "test" else label_
                examples.append(InputExample(guid=guid_, text_a=dialog, text_b=None, label=label))
        return examples

class DreamProcessor(DataProcessor):
    """Processor for the DREAM Dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, e) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            context = " ".join(e[0])
            '''
            qas = e[1][0]
            question = qas['question']
            choices = qas['choice']
            answer = qas['answer']
            for (j,choice) in enumerate(choices):
                if choice == answer:
                    label = 1
                else:
                    label = 0
                guid_=guid+"-"+str(j)
                examples.append(InputExample(guid=guid_, text_a=context, text_b=question+" "+choice,label=label))
            '''

            qas_s = e[1]
            for (t,qas) in enumerate(qas_s):
                question = qas['question']
                choices = qas['choice']
                answer = qas['answer']
                for (j,choice) in enumerate(choices):
                    if choice == answer:
                        label = 1
                    else:
                        label = 0
                    guid_=guid+"-"+str(t)+"-"+str(j)
                    examples.append(InputExample(guid=guid_, text_a=context, text_b=question+" "+choice,label=label))

        return examples

class FedTurnProcessor(DataProcessor):
    """Processor for the FED dataset -- turn-interesting ."""
    def __init__(self,rankings,aspect):
        self.rankings=rankings
        self.aspect=aspect

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return self.rankings

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, e) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            context=" ".join(e['context'].split("\n"))
            response=e['response']
            xx=e['annotations'][self.aspect]
            score=0
            count=0
            for i in xx:
                if type(i)==type(1):
                    score+=i
                    count+=1
            if count!=0:
                label=float(score)/count
            else:
                label=self.rankings[1]
            examples.append(InputExample(guid=guid, text_a=context, text_b=response, label=label))
        return examples

class FedDialogProcessor(DataProcessor):
    """Processor for the FED dataset -- turn-interesting ."""
    def __init__(self,rankings,aspect):
        self.rankings=rankings
        self.aspect=aspect

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return self.rankings

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, e) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            context=" ".join(e['context'].split("\n"))
            xx=e['annotations'][self.aspect]
            score=0
            count=0
            for i in xx:
                if type(i)==type(1):
                    score+=i
                    count+=1
            if count!=0:
                label=float(score)/count
            else:
                label=self.rankings[1]
            examples.append(InputExample(guid=guid, text_a=context, text_b=None, label=label))
        return examples

class DDPCProcessor(DataProcessor):
    """Processor for the DD/PC understandable ."""

    def __init__(self,aspect,rankings):
        self.aspect=aspect
        self.rankings=rankings

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return self.rankings

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, e) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            context = e['history']
            utt = e['utt']
            label=e[self.aspect]
            if self.aspect in ['grammar','content','overall','relevance']:
                label=label-1
            examples.append(InputExample(guid=guid, text_a=context, text_b=utt, label=label))
        return examples

class NSPProcessor(DataProcessor):

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["text"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def get_rankingss(self):
        return [0,1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["utt_1"]
            text_b = line["utt_2"]
            label = None if set_type.startswith("test") else float(line["label"])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

dialog_evaluation_tasks_num_labels = {
    "nsp_pretrain":1,
    
    "convai2_engagement":1,
    "rawtext_pretrain":1,
    "rawtext_without_nidf_pretrain":1,
    "rawtext_1_pretrain":1,
    "rawtext_2_pretrain":1,
    "rawtext_3_pretrain":1,
    "mutual":1,
    "mutual_plus":1,
    'dream':1,

    "fed_turn_interesting":1,
    "fed_turn_engaging": 1,
    "fed_turn_specific": 1,
    "fed_turn_relevant": 1,
    "fed_turn_correct": 1,
    "fed_turn_semanticallyappropriate": 1,
    "fed_turn_understandable": 1,
    "fed_turn_fluent": 1,
    "fed_turn_overall": 1,
    "fed_dialog_coherent": 1,
    "fed_dialog_errorrecovery": 1,
    "fed_dialog_consistent": 1,
    "fed_dialog_likeable": 1,
    "fed_dialog_understanding": 1,
    "fed_dialog_flexible": 1,
    "fed_dialog_informative": 1,
    "fed_dialog_inquisitive": 1,
    "fed_dialog_diverse": 1,
    "fed_dialog_depth": 1,
    "fed_dialog_overall": 1,

    "dd_content":1,
    "dd_fact":1,
    "dd_grammar":1,
    "dd_overall":1,
    "dd_relevance":1,

    "pc_content":1,
    "pc_fact":1,
    "pc_grammar":1,
    "pc_overall":1,
    "pc_relevance":1,
}

dialog_evaluation_tasks_num_rankings = {
    "nsp_pretrain":1,
    "convai2_engagement":5,
    "rawtext_pretrain":1,
    "rawtext_without_nidf_pretrain":1,
    "rawtext_1_pretrain":1,
    "rawtext_2_pretrain":1,
    "rawtext_3_pretrain":1,
    "mutual":1,
    "mutual_plus":1,
    'dream':1,

    "fed_turn_interesting":2,
    "fed_turn_engaging": 2,
    "fed_turn_specific": 2,
    "fed_turn_relevant": 2,
    "fed_turn_correct": 2,
    "fed_turn_semanticallyappropriate": 2,
    "fed_turn_understandable": 1,
    "fed_turn_fluent": 2,
    "fed_turn_overall": 4,
    "fed_dialog_coherent": 2,
    "fed_dialog_errorrecovery": 2,
    "fed_dialog_consistent": 1,
    "fed_dialog_likeable": 2,
    "fed_dialog_understanding": 2,
    "fed_dialog_flexible": 2,
    "fed_dialog_informative": 2,
    "fed_dialog_inquisitive": 2,
    "fed_dialog_diverse": 2,
    "fed_dialog_depth": 2,
    "fed_dialog_overall": 4,

    "dd_content":4,
    "dd_fact":1,
    "dd_grammar":4,
    "dd_overall":4,
    "dd_relevance":4,

    "pc_content":4,
    "pc_fact":1,
    "pc_grammar":4,
    "pc_overall":4,
    "pc_relevance":4,
}

dialog_evaluation_processors = {
    "nsp_pretrain":NSPProcessor,
    "convai2_engagement":Convai2EngagementProcessor,
    "rawtext_pretrain":RawtextProcessor,
    "rawtext_without_nidf_pretrain":RawtextWithoutNIDFProcessor,
    "rawtext_1_pretrain":Rawtext_1_Processor,
    "rawtext_2_pretrain":Rawtext_2_Processor,
    "rawtext_3_pretrain":Rawtext_3_Processor,
    "mutual":MutualProcessor,
    "mutual_plus":MutualProcessor,
    "dream":DreamProcessor,

    "fed_turn_interesting":FedTurnProcessor([0,2],'Interesting'),
    "fed_turn_engaging": FedTurnProcessor([0,2],'Engaging'),
    "fed_turn_specific": FedTurnProcessor([0,2],'Specific'),
    "fed_turn_relevant": FedTurnProcessor([0,2],'Relevant'),
    "fed_turn_correct": FedTurnProcessor([0,2],'Correct'),
    "fed_turn_semanticallyappropriate": FedTurnProcessor([0,2],'Semantically appropriate'),
    "fed_turn_understandable": FedTurnProcessor([0,1],'Understandable'),
    "fed_turn_fluent": FedTurnProcessor([0,2],'Fluent'),
    "fed_turn_overall": FedTurnProcessor([0,4],'Overall'),
    "fed_dialog_coherent": FedDialogProcessor([0,2],'Coherent'),
    "fed_dialog_errorrecovery": FedDialogProcessor([0,2],'Error recovery'),
    "fed_dialog_consistent": FedDialogProcessor([0,1],'Consistent'),
    "fed_dialog_likeable": FedDialogProcessor([0,2],'Likeable'),
    "fed_dialog_understanding": FedDialogProcessor([0,2],'Understanding'),
    "fed_dialog_flexible": FedDialogProcessor([0,2],'Flexible'),
    "fed_dialog_informative": FedDialogProcessor([0,2],'Informative'),
    "fed_dialog_inquisitive": FedDialogProcessor([0,2],'Inquisitive'),
    "fed_dialog_diverse": FedDialogProcessor([0,2],'Diverse'),
    "fed_dialog_depth": FedDialogProcessor([0,2],'Depth'),
    "fed_dialog_overall": FedDialogProcessor([0,4],'Overall'),

    "dd_content":DDPCProcessor("content",[0,4]),
    "dd_fact":DDPCProcessor("fact",[0,1]),
    "dd_grammar":DDPCProcessor("grammar",[0,4]),
    "dd_overall":DDPCProcessor("overall",[0,4]),
    "dd_relevance":DDPCProcessor("relevance",[0,4]),

    "pc_content":DDPCProcessor("content",[0,4]),
    "pc_fact":DDPCProcessor("fact",[0,1]),
    "pc_grammar":DDPCProcessor("grammar",[0,4]),
    "pc_overall":DDPCProcessor("overall",[0,4]),
    "pc_relevance":DDPCProcessor("relevance",[0,4]),
}

dialog_evaluation_output_modes = {
    "nsp_pretrain":"regression",
    "convai2_engagement": "regression",
    "rawtext_pretrain":"regression",
    "rawtext_without_nidf_pretrain":"regression",
    "rawtext_1_pretrain":"regression",
    "rawtext_2_pretrain":"regression",
    "rawtext_3_pretrain":"regression",
    "mutual":"regression",
    "mutual_plus":"regression",
    "dream":"regression",

    "fed_turn_interesting":"regression",
    "fed_turn_engaging": "regression",
    "fed_turn_specific": "regression",
    "fed_turn_relevant": "regression",
    "fed_turn_correct": "regression",
    "fed_turn_semanticallyappropriate": "regression",
    "fed_turn_understandable": "regression",
    "fed_turn_fluent": "regression",
    "fed_turn_overall": "regression",
    "fed_dialog_coherent": "regression",
    "fed_dialog_errorrecovery": "regression",
    "fed_dialog_consistent": "regression",
    "fed_dialog_likeable": "regression",
    "fed_dialog_understanding": "regression",
    "fed_dialog_flexible": "regression",
    "fed_dialog_informative": "regression",
    "fed_dialog_inquisitive": "regression",
    "fed_dialog_diverse": "regression",
    "fed_dialog_depth": "regression",
    "fed_dialog_overall": "regression",

    "dd_content":"regression",
    "dd_fact":"regression",
    "dd_grammar":"regression",
    "dd_overall":"regression",
    "dd_relevance":"regression",

    "pc_content":"regression",
    "pc_fact":"regression",
    "pc_grammar":"regression",
    "pc_overall":"regression",
    "pc_relevance":"regression",
}
