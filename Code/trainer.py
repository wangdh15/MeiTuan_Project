# -*- coding:utf-8 -*-
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import logging.handlers
import config
from utils import check_path
from data_loader import data_loader
import pandas as pd
import os
from sklearn import metrics


class trainer:

    def __init__(self, data, config):
        '''
        初始化各个变量
        :param train_X:
        :param train_Y:
        '''
        self.train_X = data['train_X']
        self.train_Y = data['train_Y']
        self.valid_X = data['valid_X']
        self.valid_Y = data['valid_Y']
        self.test_X = data['test_X']
        self.test_Y = data['test_Y']
        self.config = config
        pass


    def pre_process(self):
        '''
        对输入的训练集数据进行一些预处理，得到网络的输入
        :return:
        '''
        # TODO 在输入到训练集中时，对训练数据的一些操作
        self.train_X = self.train_X.drop(self.config.train_droped_feature, axis=1)
        self.valid_X = self.valid_X.drop(self.config.train_droped_feature, axis=1)
        self.test_X = self.test_X.drop(self.config.train_droped_feature, axis=1)




    def lgb_fit(self):
        """模型（交叉验证）训练，并返回最优迭代次数和最优的结果。
        Args:
            config: xgb 模型参数 {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
            X_train：array like, shape = n_sample * n_feature
            y_train:  shape = n_sample * 1

        Returns:
            best_model: 训练好的最优模型
            best_auc: float, 在测试集上面的 AUC 值。
            best_round: int, 最优迭代次数。
        """
        params = config.params
        max_round = config.max_round
        early_stop_round = config.early_stop_round
        seed = config.seed


        # 是否区分类型特征
        if config.categorical_feature is not None:
            dtrain = lgb.Dataset(self.train_X, label=self.train_Y, categorical_feature=config.categorical_feature)
            dvalid = lgb.Dataset(self.valid_X, label=self.valid_Y, categorical_feature=config.categorical_feature)
        else:
            dtrain = lgb.Dataset(self.train_X, label=self.train_Y)
            dvalid = lgb.Dataset(self.valid_X, label=self.valid_Y)
        watchlist = [dtrain, dvalid]

        tic = time.time()
        best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        print('Time cost {}s'.format(time.time() - tic))

        y_pred = best_model.predict(self.test_X)
        test_auc = metrics.roc_auc_score(self.test_Y, y_pred)

        best_round = best_model.best_iteration
        best_auc = best_model.best_score

        print('best_round={}, best_auc={}, test_auc={}'.format(best_round, best_auc, test_auc))
        return best_model, test_auc



    def run_cv(self, X_train, y_train, config):
        '''
        进行交叉验证
        :param X_train: 训练数据
        :param y_train: 测试数据
        :param config: 配置文件
        :return:
        '''
        tic = time.time()
        lgb_model, best_auc, best_round, cv_result = self.lgb_fit(config, X_train, y_train)
        print('Time cost {}s'.format(time.time() - tic))
        result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
        print(result_message)
        return lgb_model


    def train(self):
        '''
        统计对外的接口
        :return:
        '''
        self.pre_process()
        print("begin train")
        print('X_train.shape={}, Y_train.shape={}'.format(self.train_X.shape, self.train_Y.shape))
        print('X_valid.shape={}, Y_valid.shape={}'.format(self.valid_X.shape, self.valid_Y.shape))
        print('X_test.shape={}, Y_test.shape={}'.format(self.test_X.shape, self.test_Y.shape))
        model, test_auc = self.lgb_fit()
        model.save_model(self.config.save_model_path)
        print("model is saved to %s" % config.save_model_path)
        print("train end")
        return model, test_auc
#     lgb_predict(lgb_model, X_test, result_path)





if __name__ == '__main__':
    pass
