# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import os
import pickle
import random
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_path", "./input",
    "The model path of bert."
)

flags.DEFINE_string("input_file", "./input/train_data",
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", "./input/train.tf_record",
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string('data_config_path', "data.conf",
                    'data config file, which save train and dev config')

flags.DEFINE_integer("batch_size", 32, "Total batch size.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", True,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 64, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_labels, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.segment_labels = segment_labels
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_labels]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def read_features_from_example_files(input_files):
    """Read features from TF example files"""

    def parse_exmp(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'input_ids': tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                'input_mask': tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                'segment_ids': tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
                'masked_lm_positions': tf.VarLenFeature(tf.int64),
                'masked_lm_ids': tf.VarLenFeature(tf.int64),
                'masked_lm_weights': tf.VarLenFeature(tf.float32)
            })
        return features

    if not type(input_files) == list:
        input_files = [input_files]
    dataset = tf.data.TFRecordDataset(input_files)
    d = dataset.map(parse_exmp)
    d = d.batch(batch_size=FLAGS.batch_size, drop_remainder=False)
    return d


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    if os.path.exists(os.path.join(FLAGS.bert_path, 'label2id.pkl')):
        # 读取label->index 的map
        with codecs.open(os.path.join(FLAGS.bert_path, 'label2id.pkl'), 'rb') as reader:
            label_map = pickle.load(reader)
    else:
        label_list = []
        label_map = {}
        for instance in instances:
            label_list.extend(instance.segment_labels)
        label_list = list(set(label_list))
        # 1表示从1开始对label进行index化
        for (i, label) in enumerate(label_list, 1):
            label_map[label] = i
        # 保存label->index 的map
        with codecs.open(os.path.join(FLAGS.bert_path, 'label2id.pkl'), 'wb') as writer:
            pickle.dump(label_map, writer)

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [label_map[label] for label in instance.segment_labels]
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(label_map['O'])

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_ner_instances(input_files, tokenizer, max_seq_length,
                                  dupe_factor, masked_lm_prob,
                                  max_predictions_per_seq, rng):
    """Create `NerTrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        lines = []
        with tf.gfile.GFile(input_file, "r") as reader:
            cur_words = []
            cur_labels = []
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                word = ''
                label = ''
                line = line.strip()
                tokens = line.split(' ')
                if len(tokens) == 2:
                    word = tokens[0]
                    label = tokens[-1]
                else:
                    if len(line) == 0:
                        """为了方便解析，因此将words和labels存入同一个列表"""
                        words = ' '.join([word for word in cur_words if len(word) > 0])
                        labels = ' '.join([label for label in cur_labels if len(label) > 0])
                        lines.append([words, labels])
                        cur_words = []
                        cur_labels = []
                        continue
                if line.startswith("-DOCSTART-"):
                    cur_words.append('')
                    continue
                cur_words.append(word)
                cur_labels.append(label)
        for line in lines:
            token, label = tokenizer.tokenize(line[0], line[1])
            if token and label:
                all_documents[-1].append([token, label])
        all_documents.append([])

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.full_tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            pass
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, masked_lm_prob,
                    max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, masked_lm_prob,
        max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    # 从文章列表中选取对应索引的文章
    document = all_documents[document_index]

    instances = []
    ################################################################### Roberta的DOC-SENTENCES替换
    # for i in range(len(document)):
    #     segment = document[i]
    #     if len(segment[0]) >= max_seq_length - 1:
    #         # -2 的原因是因为序列需要加一个句首"[CLS]"和句尾"[SEP]"标志，
    #         # 其中"[CLS]"是用来作为句子分类的标志，而"[SEP]"是用来作为句子分割的标志
    #         segment = [segment[0][0:(max_seq_length - 2)], segment[1][0:(max_seq_length - 2)]]
    #
    #     # tokens记录原始字符串转换成的列表，segment_labels记录原始实体标签
    #     tokens = []
    #     segment_labels = []
    #     tokens.append("[CLS]")
    #     # 标签中空白符用“O”指代
    #     segment_labels.append("O")
    #     for index in range(len(segment[0])):
    #         tokens.append(segment[0][index])
    #         segment_labels.append(segment[1][index])
    #
    #     tokens.append("[SEP]")
    #     segment_labels.append("O")

    max_sequence_length_segments = get_max_sequence_length_segments(document, max_seq_length)
    # leng = len(max_sequence_length_segments)
    # if os.path.exists(FLAGS.data_config_path):
    #     with codecs.open(FLAGS.data_config_path) as fd:
    #         data_config = json.load(fd)
    #     data_config['num_train_size'] = leng
    # else:
    #     data_config = {'num_train_size': leng}
    # with codecs.open(FLAGS.data_config_path, 'a', encoding='utf-8') as fd:
    #     json.dump(data_config, fd)
    for segment in max_sequence_length_segments:

        # tokens记录原始字符串转换成的列表，segment_labels记录原始实体标签
        tokens = []
        segment_labels = []
        tokens.append("[CLS]")
        # 标签中空白符用“O”指代
        segment_labels.append("O")
        for index in range(len(segment[0])):
            tokens.append(segment[0][index])
            segment_labels.append(segment[1][index])

        tokens.append("[SEP]")
        segment_labels.append("O")

        ##################################################################################
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        instance = TrainingInstance(
            tokens=tokens,
            segment_labels=segment_labels,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    while True:
                        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                        if not masked_token.startswith("##"):
                            break

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index] if not tokens[index].startswith("##")
            else tokens[index][2:]))

    # 去除token前缀“##”
    for index in range(len(output_tokens)):
        if output_tokens[index].startswith("##"):
            output_tokens[index] = output_tokens[index][2:]

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def get_max_sequence_length_segments(document, max_sequence_length):  # 新增的方法
    """
    获取初步的训练实例，将整段按照max_sequence_length切分成多个部分,并以多个处理好的实例的形式返回。
    :param document: 一整段
    :param max_sequence_length:
    :return: a list. each element is a sequence of text
    """
    max_sequence_length_allowed = max_sequence_length - 2
    document = [seq for seq in document if len(seq[0]) < max_sequence_length_allowed]
    sizes = [len(seq[0]) for seq in document]

    result_list = []
    curr_seq = [[], []]  # 当前处理的序列
    sz_idx = 0
    while sz_idx < len(sizes):
        # 当前句子加上新的句子，如果长度小于最大限制，则合并当前句子和新句子；否则即超过了最大限制，那么做为一个新的序列加到目标列表中
        if not curr_seq[0] or len(curr_seq[0]) + sizes[sz_idx] <= max_sequence_length_allowed:
            curr_seq[0].extend(document[sz_idx][0])
            curr_seq[1].extend(document[sz_idx][1])
            sz_idx += 1
        else:
            result_list.append(curr_seq)
            curr_seq = [[], []]
    # 对最后一个序列进行处理，如果太短的话，丢弃掉。
    if len(curr_seq[0]) > max_sequence_length_allowed / 2:  # /2
        result_list.append(curr_seq)

    return result_list


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.NerTokenizer(
        vocab_file=FLAGS.vocab_file)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_ner_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer.full_tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


def load_features():
    """从examples files读取features"""
    features = read_features_from_example_files(FLAGS.output_file)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        iterator = features.make_one_shot_iterator()
        next_element = iterator.get_next()
        # 不断的获得下一个样本
        while True:
            try:
                # 获得的值直接属于graph的一部分，所以不再需要用feed_dict来喂
                features = sess.run(next_element)
                # input_ids = features["input_ids"][1]
                # tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)
                # input_labels = tokenizer.convert_ids_to_tokens(input_ids)
                print()
            # 如果遍历完了数据集，则返回错误
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break


if __name__ == "__main__":
    load_features()
    # flags.mark_flag_as_required("input_file")
    # flags.mark_flag_as_required("output_file")
    # flags.mark_flag_as_required("vocab_file")
    # tf.app.run()

