# config file

# class Config(object):
#     def __init__(self):
#         self.params = {
#             'objective': 'binary',
#             'metric': {'auc'},
#             'learning_rate': 0.05,
#             'num_leaves': 50,  # 叶子设置为 50 线下过拟合严重
#             'min_sum_hessian_in_leaf': 0.1,
#             'feature_fraction': 0.3,  # 相当于 colsample_bytree
#             'bagging_fraction': 0.5,  # 相当于 subsample
#             'lambda_l1': 0,
#             'lambda_l2': 5,
#             'num_thread': 13  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
#         }
#         self.max_round = 10000
#         self.cv_folds = 5
#         self.early_stop_round = 30
#         self.seed = 3
#         self.save_model_path = 'model/lgb.txt'

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
    'num_thread': 20  # 线程数设置为真实的 CPU 数，一般12线程的机器有6个物理核
}

max_round = 10000
cv_folds = 5
early_stop_round = 50
seed = 3
save_model_path = "model/lgb.txt"
original_data_dir = "../ctr_data"
preprocessed_data_dir = "../ctr_data/Processed_Feature"
save_result_path = "./result/" + str(params['num_leaves']) + ".csv"


