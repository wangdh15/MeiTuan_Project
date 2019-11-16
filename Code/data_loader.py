# -*- coding:utf-8 -*-
import os
import pandas
import numpy
import pandas as pd
import pickle
from utils import check_path
from feature_extractor.poi_feature import poi_feature_extractor
from feature_extractor.user_feature import user_feature_extractor
from feature_extractor.distance_feature import distance_feature_extractor
from feature_extractor.impr_click_action_feature import impr_click_action_feature_extractor
from feature_extractor.cate_feature import cate_feature_extractor
from feature_extractor.pos_feature import pos_feature_extractor

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
        self.pos_feature_extractor = pos_feature_extractor(self.config, self.train_origin_data)

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
        # 取出类中的原始数据，便于后面的形式都一样
        train_merged_feature = self.train_origin_data
        test_merged_feature = self.test_origin_data

        # 提取商户的团单的各种特征
        poi_feature = self.poi_feature_extractor.get_feature()
        train_merged_feature = pd.merge(train_merged_feature, poi_feature, on="poi_id", how='left')
        test_merged_feature = pd.merge(test_merged_feature, poi_feature, on="poi_id", how='left')

        # 提取训练集和测试集上的距离特征
        train_distance_feature, test_distance_feature = self.distance_feature_extractor.get_feature()
        train_merged_feature = pd.merge(train_merged_feature, train_distance_feature, on="request_id", how='left')
        test_merged_feature = pd.merge(test_merged_feature, test_distance_feature, on='ID', how="left")

        # 提取 点击的时间特征
        train_impr_click_action_feature, test_impr_click_action_feature = self.impr_click_action_feature_extractor.get_feature()
        train_merged_feature = pd.merge(train_merged_feature, train_impr_click_action_feature, on="request_id", how='left')
        test_merged_feature = pd.merge(test_merged_feature, test_impr_click_action_feature, on="ID", how='left')

        # 提取每个商户在每种cate的pos特征
        pos_cate_feature = self.pos_feature_extractor.get_feature()
        train_merged_feature = pd.merge(train_merged_feature, pos_cate_feature, on=['poi_id', 'cate_id'], how='left')
        test_merged_feature = pd.merge(test_merged_feature, pos_cate_feature, on=['poi_id', 'cate_id'], how='left')

        return train_merged_feature, test_merged_feature


    def post_process(self, train_merged_feature, test_merged_feature):
        '''
        后处理函数，对于合并完各种特征的数据而言，对需要转换类型的操作在这里实现
        :return:
        '''
        # TODO 一些后处理的工作

        train_merged_feature = train_merged_feature[train_merged_feature['action'].isin([0, 1])]
        # 数据扩增，对训练数据中的action为1的数据进行重复，使之达到正负样例比例为1：5

        # 将数据随机打乱
        train_merged_feature = train_merged_feature.sample(frac=1.0)

        # 划分出训练集，验证集和测试集
        train_merged_feature = train_merged_feature.reset_index().drop(['index'], axis=1)
        train_set = train_merged_feature[0:int(0.8*len(train_merged_feature))]
        valid_set = train_merged_feature[int(0.8*len(train_merged_feature)) : int(0.9*len(train_merged_feature))]
        test_set = train_merged_feature[int(0.9*len(train_merged_feature)) : len(train_merged_feature)]

        if self.config.data_augmentation:
            # 如果需要数据增强的话，对训练集中的label为1的数据进行重复
            train_set_1 = train_set[train_set['action'] == 1]
            train_set_0 = train_set[train_set['action'] == 0]
            train_set_1_multi = pd.concat([train_set_1] * 9)
            train_set = pd.concat([train_set_0, train_set_1_multi])
            train_set = train_set.sample(frac = 1.0)

        # 得到训练集/验证集/测试集的输入和label
        train_X = train_set.drop(['action'], axis=1)
        train_Y = train_set['action']
        valid_X = valid_set.drop(['action'], axis=1)
        valid_Y = valid_set['action']
        test_X = test_set.drop(['action'], axis=1)
        test_Y = test_set['action']

        # 需要上传的最终的测试集
        final_test_X = test_merged_feature

        # drop掉不用的特征
        train_X = train_X.drop(self.config.train_droped_feature, axis=1)
        valid_X = valid_X.drop(self.config.train_droped_feature, axis=1)
        test_X = test_X.drop(self.config.train_droped_feature, axis=1)
        final_test_X = final_test_X.drop(self.config.test_droped_feature, axis=1)

        # 判断训练和测试的输入网路的特征个数是否相同
        assert (train_X.shape[1] ) == (final_test_X.shape[1])

        # 判断测试集的数目是否发生变化
        assert final_test_X.shape[0] == self.test_origin_data.shape[0]
        print("train_X_input feature number is %d" % (train_X.shape[1]))
        print("test_X_input feature number is %d" % (final_test_X.shape[1]))

        return train_X, train_Y, valid_X, valid_Y, test_X, test_Y, final_test_X


    def get_data(self):
        '''
        data_loader类对外的唯一接口，用于提供训练集和测试集
        :return:
        '''
        # 调用merge_feature，合并各种特征
        #
        train_merged_feature, test_merged_feature = self.merge_feature()

        train_X, train_Y, valid_X, valid_Y, test_X, test_Y, final_test_X = self.post_process(train_merged_feature, test_merged_feature)

        return {'train_X':train_X,
                'train_Y':train_Y,
                'valid_X':valid_X,
                'valid_Y':valid_Y,
                'test_X':test_X,
                'test_Y':test_Y,
                'final_test_X':final_test_X}

if __name__ == '__main__':

    pass

