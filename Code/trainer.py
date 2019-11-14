import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import logging.handlers
import config
from utils import check_path
from data_loader import data_loader
import pandas as pd
import os


class trainer:

    def __init__(self, train_X, train_Y, config):
        '''
        初始化各个变量
        :param train_X:
        :param train_Y:
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.config = config
        pass


    def pre_process(self):
        '''
        对输入的训练集数据进行一些预处理，得到网络的输入
        :return:
        '''
        # TODO 在输入到训练集中时，对训练数据的一些操作
        train_X_input = self.train_X.drop(self.config.train_droped_feature, axis=1)
        return train_X_input



    def lgb_fit(self, config, X_train, y_train):
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
        cv_folds = config.cv_folds
        early_stop_round = config.early_stop_round
        seed = config.seed
        if cv_folds is not None:
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=config.categorical_feature)
            cv_result = lgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                               metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
            # 最优模型，最优迭代次数
            best_round = len(cv_result['auc-mean'])
            best_auc = cv_result['auc-mean'][-1]  # 最好的 auc 值
            best_model = lgb.train(params, dtrain, best_round, categorical_feature=config.categorical_feature)
        else:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=43)
            X_valid1, X_valid2, y_valid1, y_valid2 = train_test_split(X_valid, y_valid, test_size=0.5, random_state=88)
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=config.categorical_feature)
            # dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid1 = lgb.Dataset(X_valid1, label=y_valid1, categorical_feature=config.categorical_feature)
            # dvalid1 = lgb.Dataset(X_valid1, label=y_valid1)
            dvalid2 = lgb.Dataset(X_valid2, label=y_valid2, categorical_feature=config.categorical_feature)
            # dvalid2 = lgb.Dataset(X_valid2, label=y_valid2)
            watchlist = [dtrain, dvalid1, dvalid2]
            best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
            best_round = best_model.best_iteration
            best_auc = best_model.best_score
            cv_result = None
        return best_model, best_auc, best_round, cv_result



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
        train_X_input = self.pre_process()
        train_Y_input = self.train_Y
        print("begin train")
        print('X_train.shape={}, Y_train.shape={}'.format(train_X_input.shape, train_Y_input.shape))
        model = self.run_cv(train_X_input, train_Y_input, self.config)
        model.save_model(self.config.save_model_path)
        print("model is saved to %s"%config.save_model_path)
        print("train end")

        return model
#     lgb_predict(lgb_model, X_test, result_path)





if __name__ == '__main__':
    pass