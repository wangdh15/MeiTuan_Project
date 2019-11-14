# -*- coding:utf-8 -*-
import os
import pandas
import numpy
import pandas as pd
import config
import pickle
from utils import check_path
from feature_extractor.poi_feature import poi_feature_extractor
from feature_extractor.user_feature import user_feature_extractor
from feature_extractor.distance_feature import distance_feature_extractor
from feature_extractor.impr_click_action_feature import impr_click_action_feature_extractor
from feature_extractor.cate_feature import cate_feature_extractor

class data_loader:
    '''
    数据加载，计算feature，返回训练集的X，Y 以及测试集的X
    '''

    def __init__(self, config):
        '''
        feature_extracted 是各部分提取的不同特征，数据类型是pandas的DataFrame {feature_name:{key:data}}
        提取特征的时候利用的是拼接起来的原始训练数据
        :param config:
        '''
        self.config = config
        self.train_origin_data, self.test_origin_data = self.read_origin_data()       # 读取原始数据
        self.poi_feature_extractor = poi_feature_extractor(self.config, self.train_origin_data)   # poi特征提取器
        # self.user_feature_extractor = user_feature_extractor(self.config, self.train_origin_data)  # user特征提取器
        self.distance_feature_extractor = distance_feature_extractor(self.config, self.train_origin_data) # 距离特征提取器
        self.impr_click_action_feature_extractor = impr_click_action_feature_extractor(self.config, self.train_origin_data, self.test_origin_data)

        # self.cate_feature_extractor = cate_feature_extractor(self.config, self.train_origin_data) # cate特征提取器
#         self.read_data()


    def read_origin_data(self):
        '''
        读取原始数据,包括训练集原始数据和测试集原始数据
        :return: train_origin_file, test_origin_file
        '''
        train_origin_data = pd.read_csv(self.config.train_origin_file)
        test_origin_data = pd.read_csv(self.config.test_origin_file)
        return train_origin_data, test_origin_data

    def merge_feature(self):
        '''
        调用各个不同的特征提取器得到的特征，然后再按照对应的键merge到训练原始数据和测试原始数据中
        :return: train_merged_feature, test_merged_feature
        '''
        # 调用各个不同的特征提取器对外的唯一接口，得到提取到的特征
        poi_feature = self.poi_feature_extractor.get_feature()
        # user_feature = self.user_feature_extractor.get_feature()
        train_distance_feature, test_distance_feature = self.distance_feature_extractor.get_feature()
        # cate_history_click_rate = self.cate_feature_extractor.get_feature()

        train_impr_click_action_feature, test_impr_click_action_feature = self.impr_click_action_feature_extractor.get_feature()
        self.train_origin_data = pd.merge(self.train_origin_data, train_impr_click_action_feature, on="request_id", how='left')
        self.test_origin_data = pd.merge(self.test_origin_data, test_impr_click_action_feature, on="request_id", how='left')

        # 将各个不同的特征提取器按照其主键拼接到训练集和测试集中
        train_merged_feature = pd.merge(self.train_origin_data, poi_feature, on="poi_id", how='left')
        # train_merged_feature = pd.merge(train_merged_feature, user_feature, on="uuid", how="left")
        train_merged_feature = pd.merge(train_merged_feature, train_distance_feature, on="request_id", how='left')
        # train_merged_feature = pd.merge(train_merged_feature, cate_history_click_rate, on="cate_id", how='left')

        test_merged_feature = pd.merge(self.test_origin_data, poi_feature, on="poi_id", how='left')
        # test_merged_feature = pd.merge(test_merged_feature, user_feature, on="uuid", how="left")
        test_merged_feature = pd.merge(test_merged_feature, test_distance_feature, on='ID', how="left")
        # test_merged_feature = pd.merge(test_merged_feature, cate_history_click_rate, on="cate_id", how='left')

        return train_merged_feature, test_merged_feature


    def post_process(self, train_merged_feature, test_merged_feature):
        '''
        后处理函数，对于合并完各种特征的数据而言，对需要转换类型的操作在这里实现
        :return:
        '''
        # TODO 一些后处理的工作
        train_X = train_merged_feature.drop(['action'], axis=1)
        train_Y = train_merged_feature['action']
        test_X = test_merged_feature
        return train_X, train_Y, test_X

    def get_data(self):
        '''
        data_loader类对外的唯一接口，用于提供训练集和测试集
        :return:
        '''
        # 调用merge_feature，合并各种特征
        #
        train_merged_feature, test_merged_feature = self.merge_feature()

        train_X, train_Y, test_X = self.post_process(train_merged_feature, test_merged_feature)

        return train_X, train_Y, test_X




#         test








if __name__ == '__main__':

    pass

