# -*- coding:utf-8 -*-
import pandas as pd
import os

class pos_feature_extractor:

    def __init__(self, config, train_origin_data):
        '''
        初始化特征提取器，输入时整个工程的配置文件，以及之前将最原始的特征合并的数据
        :param config: 配置文件
        :param origin_data: 将各个原始特征合并的数据
        '''
        self.train_origin_data = train_origin_data
        self.config = config

    def pos_cate_feature(self, recompu=False):
        '''
        计算每个商户在每种cate类型中的各种特征
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.pos_cate_feature_file):
            print("load poi_deal_feature from file : %s" % self.config.pos_cate_feature_file)
            pos_cate_feature = pd.read_csv(self.config.pos_cate_feature_file)
            return pos_cate_feature
        else:
            # 用origin_data重新计算特征,并将重新计算过后的特征持久化到硬盘上
            # TODO 重原始文件中计算特征并持久化到硬盘上
            print("compute poi_deal_feature from scratch")
            pos_cate = self.train_origin_data[['poi_id','cate_id', 'pos']].groupby(['poi_id', 'cate_id'])
            # 平均值，最小值，最大值，中位数
            pos_cate_avg = pos_cate.mean()
            pos_cate_min = pos_cate.min()
            pos_cate_max = pos_cate.max()
            pos_cate_median = pos_cate.median()

            pos_cate_feature = pd.merge(pos_cate_avg, pos_cate_min, on=['poi_id', 'cate_id'], how='left')
            pos_cate_feature = pd.merge(pos_cate_feature, pos_cate_max, on=['poi_id', 'cate_id'], how='left')
            pos_cate_feature = pd.merge(pos_cate_feature, pos_cate_median, on=['poi_id', 'cate_id'], how='left')

            pos_cate_feature.to_csv(self.config.pos_cate_feature_file, index=True, header=True)
            print("pos_cate_feature is saved to %s" % self.config.pos_cate_feature_file)

            return pos_cate_feature


    def get_feature(self):
        '''
        整个类对外的接口，调用类中其余提取特征的函数，然后将各个函数提取到的特征按照poi_id拼接起来，得到poi的全部特征，并返回
        :return:
        '''
        pos_cate_feature = self.pos_cate_feature()
        # poi_history_click_rate = self.poi_history_click_rate()
        # merge
        # poi_feature = pd.merge(poi_deal_feature, poi_history_click_rate, how="inner", on="poi_id")
        return pos_cate_feature


