#!/usr/bin/env python 
# -*- coding:utf-8 -*-
'''
@author:HE
@date:2019/9/5
@name:model.py
@IDE:PyCharm
'''
# coding: utf-8
from __future__ import print_function
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
import tensorflow as tf


class BiLSTM_CRF:
    def __init__(self, embedding, labels, sequence_lengths,
                 max_seq_length, num_classes, initializers,
                 lstm_size=128, num_layers=3, is_training=True):
        """
        BiLSTM-CRF 网络
        :param embedding: Fine-tuning embedding input
        :param labels: 真实标签
        :param sequence_lengths: [batch_size] 每个batch下序列的真实长度
        :param max_seq_length: 最大序列长度
        :param num_classes: 标签数量
        :param initializers: variable init class
        :param lstm_size: LSTM的隐含元个数
        :param num_layers: RNN的层数
        :param keep_prob: droupout rate
        """
        self.embedding = embedding
        self.labels = labels
        self.sequence_lengths = sequence_lengths
        self.num_classes = num_classes
        self.initializers = initializers
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        self.keep_prob = 1.0
        if self.is_training:
            self.keep_prob = 0.5

        self.build_lstm()
        self.build_loss()
        self.build_outputs()

    def build_lstm(self):

        # 创建双向cells
        with tf.name_scope('bi_lstm'):

            def get_a_cell(lstm_size):
                return tf.nn.rnn_cell.BasicRNNCell(num_units=lstm_size)

            # 创建多层双向cells
            if self.num_layers > 1:
                cell_bw = rnn.MultiRNNCell([get_a_cell(self.lstm_size) for _ in range(self.num_layers)],
                                           state_is_tuple=True)
                cell_fw = rnn.MultiRNNCell([get_a_cell(self.lstm_size) for _ in range(self.num_layers)],
                                           state_is_tuple=True)
            else:
                cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
                cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            if self.keep_prob != 1:
                cell_bw = rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)
                cell_fw = rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
            (cell_fw_outputs, cell_bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.embedding,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)

            # 通过lstm_outputs得到概率
            seq_output = tf.concat([cell_fw_outputs, cell_bw_outputs], axis=-1)
            seq_output = tf.nn.dropout(seq_output, keep_prob=self.keep_prob)
            self.pred_input = tf.reshape(seq_output, [-1, 2 * self.lstm_size])

            with tf.variable_scope('proj'):
                proj_w = tf.get_variable("proj_w", shape=[self.lstm_size * 2, self.num_classes], dtype=tf.float32,
                                         initializer=self.initializers.xavier_initializer())
                proj_b = tf.get_variable("proj_b", shape=[self.num_classes], dtype=tf.float32,
                                         initializer=tf.zeros_initializer())
            self.pred = tf.matmul(self.pred_input, proj_w) + proj_b
            self.logits = tf.reshape(self.pred, [-1, self.max_seq_length, self.num_classes])

    def build_loss(self):
        with tf.name_scope('crf_loss'):
            log_likelihood, self.transition_params = crf.crf_log_likelihood(inputs=self.logits, tag_indices=self.labels,
                                                                            sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

    def build_outputs(self):
        with tf.name_scope('outputs'):
            self.pred_ids, _ = crf.crf_decode(potentials=self.logits, transition_params=self.transition_params,
                                              sequence_length=self.sequence_lengths)

    def rst_tuple(self):
        return self.loss, self.logits, self.transition_params, self.pred_ids
