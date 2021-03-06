# -*- coding:utf-8 -*-
import pandas as pd
import os

class impr_click_action_feature_extractor:
    '''
    包含的特征有：
    时间特征
    '''
    def __init__(self, config, train_origin_data, test_origin_data):
        '''
        初始化特征提取器，输入时整个工程的配置文件，以及之前将最原始的特征合并的数据
        :param config: 配置文件
        :param origin_data: 将各个原始特征合并的数据
        '''
        self.train_origin_data = train_origin_data
        self.test_origin_data = test_origin_data
        self.config = config

    def train_time_feature(self, recompu=False):
        '''
        对输入的origin_data进行处理，得到feature，然后将数据返回，主键是request_id
        为避免重复计算，可以将计算好的特征持久化到磁盘中，下次调用函数直接read然后返回即可,需要同时满足config中
        的路径存在而且recompu为False
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return: feature 提取的特征
        '''

        if not recompu and os.path.exists(self.config.train_time_feature_file):
            print("load train time feature from file : %s" % self.config.train_time_feature_file)
            train_time_feature = pd.read_csv(self.config.train_time_feature_file)
            return train_time_feature
        else:
            # 用origin_data重新计算特征
            print("compute train time feature from scratch")
            train_time_feature = self.train_origin_data[['request_id','time', 'request_time']].copy()
            train_time_feature['new_day'] = pd.to_datetime(train_time_feature['time'])
            train_time_feature['new_time'] = pd.to_datetime(train_time_feature['request_time'])
            train_time_feature['year'] = train_time_feature['new_day'].dt.year
            train_time_feature['month'] = train_time_feature['new_day'].dt.month
            train_time_feature['day'] = train_time_feature['new_day'].dt.day
            train_time_feature['weekofyear'] = train_time_feature['new_day'].dt.weekofyear
            train_time_feature['dayofweek'] = train_time_feature['new_day'].dt.dayofweek
            train_time_feature['hour'] = train_time_feature['new_time'].dt.hour
            train_time_feature['minute'] = train_time_feature['new_time'].dt.minute

            train_time_feature = train_time_feature.drop(['time', 'request_time'], axis=1)

            train_time_feature.to_csv(self.config.train_time_feature_file, index=False, header=True)
            print("train time feature is stored to %s" % self.config.train_time_feature_file)

            return train_time_feature

    def test_time_feature(self, recompu=False):
        '''
        对输入的origin_data进行处理，得到feature，然后将数据返回，主键是ID
        为避免重复计算，可以将计算好的特征持久化到磁盘中，下次调用函数直接read然后返回即可,需要同时满足config中
        的路径存在而且recompu为False
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return: feature 提取的特征
        '''

        if not recompu and os.path.exists(self.config.test_time_feature_file):
            print("load test time feature from file : %s" % self.config.test_time_feature_file)
            test_time_feature = pd.read_csv(self.config.test_time_feature_file)
            return test_time_feature
        else:
            # 用origin_data重新计算特征
            print("compute test time feature from scratch")
            # time 字段包含年月日 request_time字段包含时分秒
            test_time_feature = self.test_origin_data[['ID','time', 'request_time']].copy()
            test_time_feature['new_day'] = pd.to_datetime(test_time_feature['time'])
            test_time_feature['new_time'] = pd.to_datetime(test_time_feature['request_time'])
            test_time_feature['year'] = test_time_feature['new_day'].dt.year
            test_time_feature['month'] = test_time_feature['new_day'].dt.month
            test_time_feature['day'] = test_time_feature['new_day'].dt.day
            test_time_feature['weekofyear'] = test_time_feature['new_day'].dt.weekofyear
            test_time_feature['dayofweek'] = test_time_feature['new_day'].dt.dayofweek
            test_time_feature['hour'] = test_time_feature['new_time'].dt.hour
            test_time_feature['minute'] = test_time_feature['new_time'].dt.minute

            test_time_feature = test_time_feature.drop(['time', 'request_time'], axis=1)

            test_time_feature.to_csv(self.config.test_time_feature_file, index=False, header=True)
            print("test time feature is stored to %s" % self.config.test_time_feature_file)

            return test_time_feature
    def get_feature(self):
        '''
        整个类对外的接口，调用类中其余提取特征的函数，然后将各个函数提取到的特征按照user_id拼接起来，得到user的全部特征，并返回
        :return: 合并后的所有特征
        '''
        train_time_feature = self.train_time_feature()
        test_time_feature = self.test_time_feature()
        # merge
        train_impr_click_action_feature = train_time_feature
        test_impr_click_action_feature = test_time_feature
        return train_impr_click_action_feature, test_impr_click_action_feature