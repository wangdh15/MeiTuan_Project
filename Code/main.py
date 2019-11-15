<<<<<<< HEAD
# -*- coding:utf-8 -*-
import pandas as pd
import os
import numpy as np
from trainer import trainer
from tester import tester
from data_loader import data_loader
import config


def mt():
    _data_loader = data_loader(config)
    print("begin load data")
    data = _data_loader.get_data()
    _tester = tester(data, config)

    # 参数搜索
    best_auc = 0
    best_number_leaves = 0
    best_learning_rate = 0
    best_feature_fraction = 0
    for num_leaves in [20, 30, 40, 50]:
        for learning_rate in [0.025, 0.05, 0.1, 0.15, 0.20]:
            for feature_fraction in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:


                config.params['num_leaves'] = num_leaves
                config.params['learning_rate'] = learning_rate
                config.params['feature_fraction'] = feature_fraction


                print("begin train")
                _trainer = trainer(data, config)

                model, test_auc = _trainer.train()

                if test_auc > best_auc:

                    best_auc = test_auc
                    best_number_leaves = num_leaves
                    best_learning_rate = learning_rate
                    best_feature_fraction =  feature_fraction
                    best_model = model

                    print("begin test")
                    _tester.test(best_model)

    print("best para is  num_leaves:%d   learning_rate:%f  feature_fraction:%f, best_test_auc is %f" % (best_number_leaves, best_learning_rate,best_feature_fraction, best_auc))

    # general
    # print("begin train")
    # _trainer = trainer(data, config)
    #
    # model, test_auc = _trainer.train()
    #
    # print("begin test")
    # _tester = tester(data, config)
    #
    # _tester.test(model)


if __name__ == '__main__':
    mt()

=======
# -*- coding:utf-8 -*-
import pandas as pd
import os
from trainer import trainer
from tester import tester
from data_loader import data_loader
import config

def mt():
    _data_loader = data_loader(config)
    print("begin load data")
    train_X, train_Y, test_X = _data_loader.get_data()

    print("begin train")
    _trainer = trainer(train_X, train_Y, config)

    model = _trainer.train()

    print("begin test")
    _tester = tester(test_X, config)

    _tester.test(model)

if __name__ == '__main__':
    mt()

>>>>>>> a2587d06c292858265e54b24ac7ef835caccb239
