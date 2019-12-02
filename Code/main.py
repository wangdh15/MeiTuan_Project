# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
from trainer import trainer
from tester import tester
from data_loader import data_loader
import config
import random


def mt_single():
    '''
    训练单个模型
    :return:
    '''
    _data_loader = data_loader(config)
    print("begin load data")
    data = _data_loader.get_data()

    print("begin train")
    _trainer = trainer(data)

    model, test_auc = _trainer.train(config)

    print("begin test")
    _tester = tester(data)

    _tester.test(model, config, test_auc)

import logging
def grif_search():
    '''
    对参数进行网格搜索
    :param param_grid:
    :return:
    '''
    logging.basicConfig(filename="train.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
    _data_loader = data_loader(config)
    print("begin load data")
    data = _data_loader.get_data()
    _trainer = trainer(data)
    _tester = tester(data)


    best_auc = 0
    best_number_leaves = 0
    # best_learning_rate = 0
    best_feature_fraction = 0
    # outcome = []
    for num_leaves in [60, 70]:
        for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:

            config.params['num_leaves'] = num_leaves
            config.params['feature_fraction'] = feature_fraction

            print("begin train")

            model, test_auc = _trainer.train(config)
            _tester.test(model, config, test_auc)
            # outcome.append([num_leaves,feature_fraction,test_auc])
            logging.info('num_leaves=%d;feature_fraction=%f:test_auc=%f' % (num_leaves, feature_fraction, test_auc))
            if test_auc > best_auc:

                best_auc = test_auc
                best_number_leaves = num_leaves
                best_feature_fraction = feature_fraction
                best_model = model

                # print("begin test")
                # _tester.test(best_model, config)

    print("best para is  num_leaves:%d   feature_fraction:%f , best_test_auc is %f" % (best_number_leaves, best_feature_fraction, best_auc))

def grif_search2():
    '''
    对参数进行网格搜索
    :param param_grid:
    :return:
    '''
    logging.basicConfig(filename="train_multiplys.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)

    best_auc = 0
    # best_number_leaves = 0
    # best_learning_rate = 0
    best_multiplys = 0
    # outcome = []
    for multiplys in [0,8,1,2,3,4,5,6,7,8,9]:
        config.multiplys = multiplys
        _data_loader = data_loader(config)
        print("begin load data")
        data = _data_loader.get_data()
        _trainer = trainer(data)
        _tester = tester(data)
        print("begin train")

        model, test_auc = _trainer.train(config)
        _tester.test(model, config, test_auc)
        # outcome.append([num_leaves,feature_fraction,test_auc])
        logging.info('multiplys=%d;test_auc=%f' % (multiplys, test_auc))
        if test_auc > best_auc:

            best_auc = test_auc
            best_multiplys = multiplys
            best_model = model

            # print("begin test")
            # _tester.test(best_model, config)

    print("best para is  multiplys:%d   , best_test_auc is %f" % (best_multiplys, best_auc))

if __name__ == '__main__':
    random.seed(config.seed)
    np.random.seed(config.seed)
    # mt_single()
    # grif_search()
    grif_search2()