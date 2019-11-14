
# trainer and tester
params = {
    'objective': 'binary',
    'metric': {'auc'},
    'learning_rate': 0.05,
    'num_leaves': 30,  # 叶子设置为 50 线下过拟合严重
    'min_sum_hessian_in_leaf': 0.1,
    'feature_fraction': 0.5,  # 相当于 colsample_bytree
    'bagging_fraction': 0.5,  # 相当于 subsample
    'lambda_l1': 0,
    'lambda_l2': 5,
    # "device" : "gpu"
    'num_thread': 20  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
}

max_round = 10000
cv_folds = 5
early_stop_round = 30
seed = 5
save_model_path = "./lgm.txt"
train_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type", 'pos']
categorical_feature = ['gender', 'job', 'cate_level1', 'cate_level2', 'cate_level3', 'area_id']







# tester
test_result_file = "./result_upload.csv"
test_droped_feature = ["poi_id", "uuid", "request_id", "time", "request_time", "device_type"]

# data_loader
# train_origin_file = "../ctr_data/Processed_Feature/train_origin_data.csv"
# test_origin_file = "../ctr_data/Processed_Feature/test_origin_data.csv"
train_origin_file = "../ctr_data/Processed_Data_2/train_plain.csv"
test_origin_file = "../ctr_data/Processed_Data_2/test_plain.csv"

# poi_feature_extractor



# user_feature_extractor





