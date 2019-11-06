import os
import pandas
import numpy
import pandas as pd
import config
import pickle
from utils import check_path

class data_loader:
    '''
    数据加载，计算feature，返回训练集的X，Y 以及测试集的X
    '''

    def __init__(self, config):
        '''
        feature_extracted 是各部分提取的不同特征，数据类型是pandas的DataFrame {feature_name:{key:data}}
        :param config:
        '''
        self.data_dir = config.original_data_dir
        check_path(self.data_dir)
        self.feature_extracted = dict()
        self.processed_data_dir = config.preprocessed_data_dir
        check_path(self.data_dir)
#         self.read_data()


    def read_data(self):
        '''
        读取原始数据
        :return:
        '''
        file_list = os.listdir(self.data_dir)
        file_list = [file for file in file_list if file.endswith("csv")]
        self.poi = pd.read_csv(os.path.join(self.data_dir, "table_poi_detail.csv"))
        self.uuid = pd.read_csv(os.path.join(self.data_dir, "table_uuid_detail.csv"))
        self.train_data = pd.read_csv(os.path.join(self.data_dir, "table_impr_click_action_train.csv"))
        self.test_data = pd.read_csv(os.path.join(self.data_dir, "table_impr_click_action_test.csv"))
        self.request_data = pd.read_csv(os.path.join(self.data_dir, "table_request_detail.csv"))
        self.deal = pd.read_csv(os.path.join(self.data_dir, "table_deal_detail.csv"))

        result = dict()
        result['poi'] = self.poi
        result["uuid"] = self.uuid
        result['train_data'] = self.train_data
        result['test_data'] = self.test_data
        result['request_data'] = self.request_data
        result['deal'] = self.deal

        return result

    def merge_feature(self):
        '''
        合并所有提取到的feature和原始feature,返回处理的X_train, Y_train, X_test
        :return:
        '''

        train_origin = self.get_origin_traindata()
        test_origin = self.get_origin_testdata()
        # 将抽取到的特征全部拼接到训练数据
        for feature in self.feature_extracted.keys():
            train_origin = pd.merge(train_origin, self.feature_extracted[feature]['data'], how="left", on=self.feature_extracted[feature]['key'])
            test_origin = pd.merge(test_origin, self.feature_extracted[feature]['data'], how="left", on=self.feature_extracted[feature]['key'])

        return {"X_train":train_origin.drop(['action'], axis=1), "Y_train":train_origin['action'], "X_test":test_origin}


    def get_origin_traindata(self, re_compute=False):
        '''
        将原始数据中的各个信息拼接起来
        :return:
        '''
        if re_compute or not os.path.exists(os.path.join(self.processed_data_dir, "train_origin_data.csv")):
            print("compute train_origin data from raw data")
            # 将用户的个人信息拼接到request表上
            user_info = pd.merge(self.request_data, self.uuid, how="left", on="uuid")

            # 测试集没有pos这个特征，所以去掉
            train_drop_pos = self.train_data.drop(["pos"], axis=1)

            # 将request表格拼接扫train表格上
            train_join_user = pd.merge(train_drop_pos, user_info, how="left", on="request_id")

            # 将poi表格拼接到train表格上
            train_join_user_poi = pd.merge(train_join_user, self.poi, how="left", on="poi_id")

            train_origin = train_join_user_poi
            
            train_origin.to_csv(os.path.join(self.processed_data_dir, 'train_origin_data.csv'),index=False,header=True)
            
#             with open(os.path.join(self.processed_data_dir, "train_origin_data"), 'wb') as f:
#                 pickle.dump(train_origin, f)

        else:
            print("load train process data from file")
#             with open(os.path.join(self.processed_data_dir, "train_origin_data"), 'rb') as f:
            train_origin = pd.read_csv(os.path.join(self.processed_data_dir, 'train_origin_data.csv'))

        return train_origin

    def get_origin_testdata(self, re_compute=False):
        '''
        将测试数据中的原始特征拼接起来
        :param re_compute:
        :return:
        '''
        # 去除测测试集中的ID字段
        if re_compute or not os.path.exists(os.path.join(self.processed_data_dir, "test_origin_data.csv")):
            print("compute process test data from scratch")
            user_info = pd.merge(self.request_data, self.uuid, how="left", on="uuid")
            test_drop_id = self.test_data.drop(['ID'], axis=1)
            test_join_user = pd.merge(test_drop_id, user_info, how="left", on="request_id")
            test_join_user_poi = pd.merge(test_join_user, self.poi, how='left', on='poi_id')
            test_origin = test_join_user_poi
#             with open(os.path.join(self.processed_data_dir, "test_origin_data"), 'wb') as f:
#                 pickle.dump(test_origin, f)
            test_origin.to_csv(os.path.join(self.processed_data_dir, 'test_origin_data.csv'),index=False,header=True)
        else:
#             with open(os.path.join(self.processed_data_dir, "test_origin_data"), 'rb') as f:
#                 test_origin = pickle.load(f)
            print("load process test data from file")
            test_origin = pd.read_csv(os.path.join(self.processed_data_dir, 'test_origin_data.csv'))

        return test_origin

if __name__ == '__main__':

    a = data_loader(config)
    result = a.merge_feature()
    # print(result)

