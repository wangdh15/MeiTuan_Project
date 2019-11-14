# -*- coding:utf-8 -*-
import pandas as pd
import os

class cate_feature_extractor:
    '''
    每一个cate的特征提取器，每一个cate的点击率等
    '''
    def __init__(self, config, origin_data):
        '''
        初始化特征提取器，输入时整个工程的配置文件，以及之前将最原始的特征合并的数据
        :param config: 配置文件
        :param origin_data: 将各个原始特征合并的数据
        '''
        self.origin_data = origin_data
        self.config = config

    def cate_history_click_rate(self, recompu=False):
        '''
        加载训练集中的距离特征
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.cate_history_click_rate_file):
            print("load cate_history_click_rate from file")
            cate_history_click_rate = pd.read_csv(self.config.cate_history_click_rate_file)
            return cate_history_click_rate
        else:
            # 用origin_data重新计算特征,并将重新计算过后的特征持久化到硬盘上
            # TODO 重原始文件中计算特征并持久化到硬盘上
            print("compute cate_history_click_rate from scratch")
            cate_history_click_rate = None
            return cate_history_click_rate

    def get_feature(self):
        '''
        整个类对外的接口，调用类中其余提取特征的函数，然后将各个函数提取到的特征按照poi_id拼接起来，得到poi的全部特征，并返回
        :return:
        '''
        cate_history_click_rate = self.cate_history_click_rate()
        # merge

        return cate_history_click_rate


