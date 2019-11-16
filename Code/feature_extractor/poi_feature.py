# -*- coding:utf-8 -*-
import pandas as pd
import os

class poi_feature_extractor:

    def __init__(self, config, origin_data):
        '''
        初始化特征提取器，输入时整个工程的配置文件，以及之前将最原始的特征合并的数据
        :param config: 配置文件
        :param origin_data: 将各个原始特征合并的数据
        '''
        self.origin_data = origin_data
        self.config = config

    def poi_deal_feature(self, recompu=False):
        '''
        计算每个商户的团单的特征
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.poi_deal_feature_file):
            print("load poi_deal_feature from file : %s" % self.config.poi_deal_feature_file)
            poi_deal_feature = pd.read_csv(self.config.poi_deal_feature_file)
            return poi_deal_feature
        else:
            # 用origin_data重新计算特征,并将重新计算过后的特征持久化到硬盘上
            # TODO 重原始文件中计算特征并持久化到硬盘上
            print("compute poi_deal_feature from scratch")
            poi_deal_feature = None
            print("poi_deal_feature is stored to %s" % self.config.poi_deal_feature_file)
            return poi_deal_feature

    def poi_history_click_rate(self, recompu=False):
        '''
        每个商户历史上被点击的概率
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.poi_history_click_rate_file):
            print("load poi_history_click_rate from file : %s" % self.config.poi_history_click_rate_file)
            poi_history_click_rate = pd.read_csv(self.config.poi_history_click_rate_file)
            return poi_history_click_rate
        else:
            # 用origin_data重新计算特征
            # TODO 重原始文件中得到每个商户历史上被点击的概率并持久化到硬盘上
            print("compute poi_history_click_rate from scratch")
            poi_history_click_rate = None
            print("poi_history_click_rate is stored to %s" % self.config.poi_history_click_rate_file)
            return poi_history_click_rate


    def get_feature(self):
        '''
        整个类对外的接口，调用类中其余提取特征的函数，然后将各个函数提取到的特征按照poi_id拼接起来，得到poi的全部特征，并返回
        :return:
        '''
        poi_deal_feature = self.poi_deal_feature()
        # poi_history_click_rate = self.poi_history_click_rate()
        # merge
        # poi_feature = pd.merge(poi_deal_feature, poi_history_click_rate, how="inner", on="poi_id")
        return poi_deal_feature


