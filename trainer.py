import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import logging.handlers
import config
from utils import check_path
from data_loader import data_loader
import pandas as pd
import os


def lgb_predict(model, X_test, config, save_result_path=None):
    print("begin predict")
    y_pred_prob = model.predict(X_test)
    if save_result_path:
        df_result = X_test
        df_result['action'] = y_pred_prob
        df_result.to_csv(save_result_path, index=False)
        print('Save the result to {}'.format(save_result_path))
#     with open(os.path.join(original_data_dir, "table_impr_click_action_test.csv"))
    test_origin = pd.read_csv(os.path.join(config.original_data_dir, "table_impr_click_action_test.csv"))
    test_origin['action'] = df_result['action']
    result_upload = test_origin[["ID", "action"]]
    # result_upload = pd.merge(test_origin, df_result, how="left", on="request_id")["ID", "action"]
    result_upload.to_csv(config.save_result_path,index=False,header=True)
    return y_pred_prob


def lgb_fit(config, X_train, y_train):
    """模型（交叉验证）训练，并返回最优迭代次数和最优的结果。
    Args:
        config: xgb 模型参数 {params, max_round, cv_folds, early_stop_round, seed, save_model_path}
        X_train：array like, shape = n_sample * n_feature
        y_train:  shape = n_sample * 1

    Returns:
        best_model: 训练好的最优模型
        best_auc: float, 在测试集上面的 AUC 值。
        best_round: int, 最优迭代次数。
    """
    params = config.params
    max_round = config.max_round
    cv_folds = config.cv_folds
    early_stop_round = config.early_stop_round
    seed = config.seed
    # seed = np.random.randint(0, 10000)
    save_model_path = config.save_model_path
    if cv_folds is not None:
        dtrain = lgb.Dataset(X_train, label=y_train)
        cv_result = lgb.cv(params, dtrain, max_round, nfold=cv_folds, seed=seed, verbose_eval=True,
                           metrics='auc', early_stopping_rounds=early_stop_round, show_stdv=False)
        # 最优模型，最优迭代次数
        best_round = len(cv_result['auc-mean'])
        best_auc = cv_result['auc-mean'][-1]  # 最好的 auc 值
        best_model = lgb.train(params, dtrain, best_round)
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=100)
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        watchlist = [dtrain, dvalid]
        best_model = lgb.train(params, dtrain, max_round, valid_sets=watchlist, early_stopping_rounds=early_stop_round)
        best_round = best_model.best_iteration
        best_auc = best_model.best_score
        cv_result = None
    if save_model_path:
        check_path(save_model_path)
        best_model.save_model(save_model_path)
    return best_model, best_auc, best_round, cv_result


# def lgb_predict(model, X_test, save_result_path=None):
#     y_pred_prob = model.predict(X_test)
#     if save_result_path:
#         df_result = df_future_test
#         df_result['orderType'] = y_pred_prob
#         df_result.to_csv(save_result_path, index=False)
#         print('Save the result to {}'.format(save_result_path))
#     return y_pred_prob



def run_cv(X_train, y_train, X_test, config):
    # train model
    tic = time.time()
    data_message = 'X_train.shape={}'.format(X_train.shape)
    print(data_message)
    logger.info(data_message)
    lgb_model, best_auc, best_round, cv_result = lgb_fit(config, X_train, y_train)
    print('Time cost {}s'.format(time.time() - tic))
    result_message = 'best_round={}, best_auc={}'.format(best_round, best_auc)
    logger.info(result_message)
    print(result_message)
    # predict
    # lgb_model = lgb.Booster(model_file=config.save_model_path)
    now = time.strftime("%m%d-%H%M%S")
    result_path = 'result/result_lgb_{}-{:.4f}.csv'.format(now, best_auc)
    check_path(result_path)
    lgb_predict(lgb_model, X_test, config, result_path)


#     lgb_predict(lgb_model, X_test, result_path)


if __name__ == '__main__':

    LOG_FILE = 'log/lgb_train.log'
    check_path(LOG_FILE)
    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=1)  # 实例化handler
    fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger('train')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    print("compute feature begin")
    data = data_loader(config).merge_feature()
    print("compute feature end")
    print("begin train")

    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test'].drop(
        ["poi_id", "uuid", "request_id", "time", "request_time", "device_type"], axis=1)
    X_train = X_train.drop(
        ["poi_id", "uuid", "request_id", "time", "request_time", "device_type"], axis=1)
    # y_train.isnull().any()
    # X_test = test_data
    data_message = 'X_train.shape={}, Y_train.shape={}'.format(X_train.shape, Y_train.shape)
    # data_message = 'X_train.shape={}, X_test.shape={}'.format(X_train.shape, X_test.shape)
    print(data_message)
    logger.info(data_message)
    # print(X_train)
    # X_train.columns = [str(x) for x in list(range(X_train.shape[1]))]
    # X_test.columns = [str(x) for x in list(range(X_train.shape[1]))]
    run_cv(X_train, Y_train, X_test, config)
    print("All Done")
