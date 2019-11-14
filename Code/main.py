import pandas
import os



from trainer import trainer
from tester import tester
from data_loader import data_loader
import config



def mt():
    _data_loader = data_loader(config)
    print("begin load data")
    train_X, train_Y, test_X = _data_loader.get_data()

    print("begin train")
    _trainer = trainer(train_X, train_Y, config)

    model = _trainer.train()

    print("begin test")
    _tester = tester(test_X, config)

    _tester.test(model)

if __name__ == '__main__':
    mt()

