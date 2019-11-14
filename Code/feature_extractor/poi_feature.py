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

    def feature_1(self, recompu=False):
        '''
        对输入的origin_data进行处理，得到feature_1，然后将数据返回，主键是poi_id
        为避免重复计算，可以将计算好的特征持久化到磁盘中，下次调用函数直接read然后返回即可,需要同时满足config中
        的路径存在而且recompu为False
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.feature_1_file_path):
            feature_1 = pd.read_csv(self.config.feature_1_file_path)
            return feature_1
        else:
            # 用origin_data重新计算特征
            feature_1 = None
            return feature_1

    def feature_2(self, recompu=False):
        '''
        对输入的origin_data进行处理，得到feature_1，然后将数据返回，主键是poi_id
        为避免重复计算，可以将计算好的特征持久化到磁盘中，下次调用函数直接read然后返回即可,需要同时满足config中
        的路径存在而且recompu为False
        :param config 整个系统的配置文件，用于从其中得到对应的特征存储的路径
        :param recompu 是否重新计算特征
        :return:
        '''

        if not recompu and os.path.exists(self.config.feature_2_file_path):
            feature_2 = pd.read_csv(self.config.feature_2_file_path)
            return feature_2
        else:
            # 用origin_data重新计算特征
            feature_2 = None
            return feature_2


    def get_feature(self):
        '''
        整个类对外的接口，调用类中其余提取特征的函数，然后将各个函数提取到的特征按照poi_id拼接起来，得到poi的全部特征，并返回
        :return:
        '''
        poi_feature = None
        feature_1 = self.feature_1()
        feature_2 = self.feature_2()
        # merge
        return poi_feature


