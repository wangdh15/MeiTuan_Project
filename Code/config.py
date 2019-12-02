# -*- coding:utf-8 -*-
# trainer
seed = 0
params = {
    'objective': 'binary',
    'metric': {'auc'},
    'learning_rate': 0.025,
    'num_leaves': 60,  # 叶子设置为 50 线下过拟合严重
    # 'max_depth':6,
    # 'min_data_in_leaf':51,
    # 'max_bin in':15,
    'min_sum_hessian_in_leaf': 0.1,
    'feature_fraction': 0.5,  # 相当于 colsample_bytree
    'bagging_fraction': 0.6,  # 相当于 subsample
    # 'bagging_freq': 0,
    'lambda_l1': 0,
    'lambda_l2': 5,
    # 'min_split_gain': 0.0,
    # "device" : "gpu"
    'num_thread': 30,  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
    # 'boosting': 'dart',
    'seed': seed
}

max_round = 3000
# # cv_folds = 5
# cv_folds = None
early_stop_round = 30


# save_model_path = "../Model/lgb_" + str(params['feature_fraction']) + "_" + str(params['bagging_fraction']) + ".txt"
save_model_path = "../Model/lgb_"  + ".txt"
train_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", 'pos',
                        'new_day','new_time', 'year', 'month',
                        'pos_cate_min', 'pos_cate_max', 'pos_cate_median']

# tester
test_result_file = "../Result/"+str(seed) + ".csv"
test_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", "ID",
                       'new_day', 'new_time', 'year', 'month',
                       'pos_cate_min', 'pos_cate_max', 'pos_cate_median']


# trainer and tester
# categorical_feature = ['gender', 'job', 'cate_level1',  'area_id']
categorical_feature = None

# data_loader
# train_origin_file = "/data1/huxiao/datasets/ctr_dataProcessed_Feature/train_origin_data.csv"
# test_origin_file = "/data1/huxiao/datasets/ctr_dataProcessed_Feature/test_origin_data.csv"
train_origin_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/train_plain.csv"
test_origin_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/test_plain.csv"
# data_augmentation = False
data_augmentation = True
# neg_pos_fraction = 5
multiplys = 8


#impr_click_action_feature
train_time_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/train_time_feature.csv"
test_time_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/test_time_feature.csv"

# poi_feature_extractor
poi_deal_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/poi_deal_feature.csv"
poi_history_click_rate_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/poi_history_click_rate.csv"



# user_feature_extractor


# distance_feature_extractor
train_distance_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/train_distance_feature.csv"
test_distance_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/test_distance_feature.csv"

# cate_feature_extractor
cate_history_click_rate_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/cate_history_click_rate.csv"

# pos_cate_feature_extractor
pos_cate_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/pos_cate_feature.csv"

# device_int_feature
device_int_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/device_int_feature.csv"

# poi_cate_click_rate
poi_cate_click_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/poi_cate_click_feature.csv"

# zy feature
train_zy_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/train_zy_feature.csv"
test_zy_feature_file = "/data1/huxiao/datasets/ctr_data/Processed_Data_2/test_zy_feature.csv"