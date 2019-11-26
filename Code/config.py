# -*- coding:utf-8 -*-
# trainer
params = {
    'objective': 'binary',
    'metric': {'auc'},
    'learning_rate': 0.025,
    'num_leaves': 40,  # 叶子设置为 50 线下过拟合严重
    'min_sum_hessian_in_leaf': 0.1,
    'feature_fraction': 0.2,  # 相当于 colsample_bytree
    'bagging_fraction': 0.5,  # 相当于 subsample
    'lambda_l1': 0,
    'lambda_l2': 5,
    # "device" : "gpu"
    'num_thread': 1  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
}

max_round = 10000
# # cv_folds = 5
# cv_folds = None
early_stop_round = 30
seed = 5

save_model_path = "../Model/lgb_" + str(params['feature_fraction']) + "_" + str(params['bagging_fraction']) + ".txt"
train_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", 'pos',
                        'new_day','new_time', 'year', 'month',
                        'pos_cate_min', 'pos_cate_max', 'pos_cate_median']

# tester
test_result_file = "../Result/add_zy_feature.csv"
test_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", "ID",
                       'new_day', 'new_time', 'year', 'month',
                       'pos_cate_min', 'pos_cate_max', 'pos_cate_median']


# trainer and tester
# categorical_feature = ['gender', 'job', 'cate_level1',  'area_id']
categorical_feature = None

# data_loader
# train_origin_file = "../ctr_data/Processed_Feature/train_origin_data.csv"
# test_origin_file = "../ctr_data/Processed_Feature/test_origin_data.csv"
train_origin_file = "../ctr_data/Processed_Data_2/train_plain.csv"
test_origin_file = "../ctr_data/Processed_Data_2/test_plain.csv"
# data_augmentation = False
data_augmentation = True
neg_pos_fraction = 5


#impr_click_action_feature
train_time_feature_file = "../ctr_data/Processed_Data_2/train_time_feature.csv"
test_time_feature_file = "../ctr_data/Processed_Data_2/test_time_feature.csv"

# poi_feature_extractor
poi_deal_feature_file = "../ctr_data/Processed_Data_2/poi_deal_feature.csv"
poi_history_click_rate_file = "../ctr_data/Processed_Data_2/poi_history_click_rate.csv"



# user_feature_extractor


# distance_feature_extractor
train_distance_feature_file = "../ctr_data/Processed_Data_2/train_distance_feature.csv"
test_distance_feature_file = "../ctr_data/Processed_Data_2/test_distance_feature.csv"

# cate_feature_extractor
cate_history_click_rate_file = "../ctr_data/Processed_Data_2/cate_history_click_rate.csv"

# pos_cate_feature_extractor
pos_cate_feature_file = "../ctr_data/Processed_Data_2/pos_cate_feature.csv"

# device_int_feature
device_int_feature_file = "../ctr_data/Processed_Data_2/device_int_feature.csv"

# poi_cate_click_rate
poi_cate_click_feature_file = "../ctr_data/Processed_Data_2/poi_cate_click_feature.csv"

# zy feature
train_zy_feature_file = "../ctr_data/Processed_Data_2/train_zy_feature.csv"
test_zy_feature_file = "../ctr_data/Processed_Data_2/test_zy_feature.csv"