# -*- coding:utf-8 -*-
import pandas as pd
import os

class distance_feature_extractor:
    '''
    距离特征提取器，提取店铺和request请求之间的距离
    '''
    def __init__(self, config, origin_data):
        '''
        初始化特征提取器，输入时整个工程的配置文件，以及之前将最原始的特征合并的数据
        :param config: 配置文件
        :param origin_data: 将各个原始特征合并的数据
        '''
        self.origin_data = origin_data
        self.config = config

    def train_distance_feature(self, recompu=False):
        '''
        加载训练集中的距离特征
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.train_distance_feature_file):
            print("load train_distance_feature from file")
            train_distance_feature = pd.read_csv(self.config.train_distance_feature_file)
            return train_distance_feature
        else:
            # 用origin_data重新计算特征,并将重新计算过后的特征持久化到硬盘上
            # TODO 重原始文件中计算特征并持久化到硬盘上
            print("compute train_distance_feature from scratch")
            train_distance_feature = None
            return train_distance_feature

    def test_distance_feature(self, recompu=False):
        '''
        每个商户历史上被点击的概率
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.test_distance_feature_file):
            print("load test_distance_feature form file")
            test_distance_feature = pd.read_csv(self.config.test_distance_feature_file)
            return test_distance_feature
        else:
            # 用origin_data重新计算特征
            # TODO 重原始文件中得到每个商户历史上被点击的概率并持久化到硬盘上
            print("compute test_distance_feature from scratch")
            test_distance_feature = None
            return test_distance_feature


    def get_feature(self):
        '''
        整个类对外的接口，调用类中其余提取特征的函数，然后将各个函数提取到的特征按照poi_id拼接起来，得到poi的全部特征，并返回
        :return:
        '''
        train_distance_feature = self.train_distance_feature()
        test_distance_feature = self.test_distance_feature()
        # merge

        return train_distance_feature, test_distance_feature


