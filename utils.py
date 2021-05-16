#!/usr/bin/env python  
#-*- coding:utf-8 -*- 

import pandas as pd
import tensorflow as tf
import random as rn
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, roc_auc_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
import logging


def seed_everything(seed):
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def recall_at_precision10(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    try:
        idx = precision.tolist().index(0.1 + np.min(abs(precision - 0.1)))
    except:
        idx = precision.tolist().index(0.1 - np.min(abs(precision - 0.1)))
    return recall[idx]






