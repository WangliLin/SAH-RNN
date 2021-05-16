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
    parser.add_argument('--X_path', default='X_test.pkl', type=str, help="the path of X_test")
    parser.add_argument('--S_path', default='S_test.pkl', type=str, help="the path of S_test")
    parser.add_argument('--y_path', default='y_test.pkl', type=str, help="the path of y_test")
    parser.add_argument('--weights_path', default='sahlstm_weights.hdf5', type=str, help="weights path")

    args = parser.parse_args()

    X_path = args.root_path + args.X_path
    S_path = args.root_path + args.S_path
    y_path = args.root_path + args.y_path
    weights_path = args.root_path + args.weights_path

    with open(X_path, 'rb') as f:
        X_test = pickle.load(f)

    with open(S_path, 'rb') as f:
        S_test = pickle.load(f)

    with open(y_path, 'rb') as f:
        y_test = pickle.load(f)

    model = SAHLSTM(granularity_num=args.granularity_num,
                    maxlen=args.maxlen,
                    max_behaviors=args.max_behaviors,
                    embedding_dims=args.embedding_dims,
                    units=args.units,
                    head_num=args.head_num,
                    ff_units=args.ff_units).get_model()

    model.load_weights(weights_path)
    logging.info(model.summary())
    y_predict = model.predict([X_test, S_test], batch_size=1024, verbose=1)
    logging.info('auc: {}'.format(roc_auc_score(y_test, y_predict)))
    logging.info('R@P=0.1: {}'.format(recall_at_precision10(y_test, y_predict)))
