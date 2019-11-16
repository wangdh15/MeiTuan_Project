# -*- coding:utf-8 -*-

class tester:

    def __init__(self, data, config):
        '''
        初始化测试器
        :param config:
        :param test_X:
        '''
        self.config = config
        self.test_X = data['final_test_X']

    def pre_process(self):
        '''
        对测试集中的数据进行一些前处理，如drop调某些列等等
        :return:
        '''
        # TODO 在网络输入以前对测试集的一些预处理
        self.test_X_input = self.test_X.drop(self.config.test_droped_feature, axis=1)

    def test(self, model):
        '''
        测试过程，调用模型，并处理得到结果
        :param model: 来自trainer返回的训练好的模型
        :param test_X:
        :param config:
        :return:
        '''
        # 对测试数据进行处理，得到网络的输入数据
        self.pre_process()

        print("begin test.....")
        print('X_test.shape={}'.format(self.test_X_input.shape))
        y_pred_prob = model.predict(self.test_X_input)
        self.test_X['action'] = y_pred_prob
        self.test_X[['ID', 'action']].to_csv(self.config.test_result_file,index=False,header=True)
        print("test Done")
        print("test result is saved to %s"%self.config.test_result_file)
        return self.test_X[['ID', 'action']]
