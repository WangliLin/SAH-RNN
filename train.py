#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from models import *
from utils import *
import pickle
import argparse


def train_model(model,
                X, y,
                weights_path,
                log_path,
                batch_size=512,
                epochs=200,
                validation_split=0.1):

    callbacks_list = [EarlyStopping(monitor='val_loss',
                                    patience=10,
                                    verbose=1,
                                    min_delta=1e-4,
                                    mode='min'),
                      ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        epsilon=1e-4,
                                        mode='min'),
                      ModelCheckpoint(monitor='val_loss',
                                      filepath=weights_path,
                                      save_best_only=True,
                                      mode='min'),
                      CSVLogger(log_path, append=True, separator=';')]

    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,
              callbacks=callbacks_list)

    return model


if __name__ == '__main__':

    seed_everything(1024)

    parser = argparse.ArgumentParser(description="Structure-Aware Hierarchical LSTM")
    parser.add_argument('--granularity_num', default=3, type=int, help="the granularity number of behavior sequence")
    parser.add_argument('--maxlen', default=500, type=int, help="the max length of behavior sequence")
    parser.add_argument('--max_behaviors', default=[358, 267, 69], type=int, nargs='+',
                        help="the number of types of behaviors over different granularities")
    parser.add_argument('--embedding_dims', default=32, type=int, help="the dimension of embedding")
    parser.add_argument('--units', default=32, type=int, help="the dimension of units in HRNN")
    parser.add_argument('--head_num', default=4, type=int, help="the number of multi-head in self-attention")
    parser.add_argument('--ff_units', default=32, type=int, help="the number of units in FeedForward Network")
    parser.add_argument('--root_path', default='data/', type=str, help="the root path")
    parser.add_argument('--X_path', default='X_train.pkl', type=str, help="the path of X_train")
    parser.add_argument('--S_path', default='S_train.pkl', type=str, help="the path of S_train")
    parser.add_argument('--y_path', default='y_train.pkl', type=str, help="the path of y_train")
    parser.add_argument('--weights_path', default='sahlstm_weights.hdf5', type=str, help="weights path")
    parser.add_argument('--log_path', default='sahlstm_log.csv', type=str, help="log path")

    args = parser.parse_args()

    X_path = args.root_path + args.X_path
    S_path = args.root_path + args.S_path
    y_path = args.root_path + args.y_path
    weights_path = args.root_path + args.weights_path
    log_path = args.root_path + args.log_path

    with open(X_path, 'rb') as f:
        X = pickle.load(f)

    with open(S_path, 'rb') as f:
        S = pickle.load(f)

    with open(y_path, 'rb') as f:
        y = pickle.load(f)

    model = SAHLSTM(granularity_num=args.granularity_num,
                    maxlen=args.maxlen,
                    max_behaviors=args.max_behaviors,
                    embedding_dims=args.embedding_dims,
                    units=args.units,
                    head_num=args.head_num,
                    ff_units=args.ff_units).get_model()

    logging.info(model.summary())

    model = train_model(model, X=[X, S], y=y, weights_path=weights_path, log_path=log_path)

    logging.info('done')



