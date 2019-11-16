# -*- coding:utf-8 -*-
# trainer
params = {
    'objective': 'binary',
    'metric': {'auc'},
    'learning_rate': 0.05,
    'num_leaves': 30,  # 叶子设置为 50 线下过拟合严重
    'min_sum_hessian_in_leaf': 0.1,
    'feature_fraction': 0.3,  # 相当于 colsample_bytree
    'bagging_fraction': 0.5,  # 相当于 subsample
    'lambda_l1': 0,
    'lambda_l2': 5,
    # "device" : "gpu"
    'num_thread': 30  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
}

max_round = 10
# # cv_folds = 5
# cv_folds = None
early_stop_round = 30
seed = 5

save_model_path = "../Model/lgb_" + str(params['feature_fraction']) + "_" + str(params['bagging_fraction']) + ".txt"

train_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", 'pos',
                        'new_day','new_time']

# tester
test_result_file = "../Result/result_upload_para_search.csv"
test_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", "ID",
                       'new_day', 'new_time']


# trainer and tester
# categorical_feature = ['gender', 'job', 'cate_level1', 'cate_level2', 'cate_level3', 'area_id']
categorical_feature = None

# data_loader
# train_origin_file = "../ctr_data/Processed_Feature/train_origin_data.csv"
# test_origin_file = "../ctr_data/Processed_Feature/test_origin_data.csv"
train_origin_file = "../ctr_data/Processed_Data_2/train_plain.csv"
test_origin_file = "../ctr_data/Processed_Data_2/test_plain.csv"
# data_augmentation = False
data_augmentation = True


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
