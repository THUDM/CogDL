from torch.utils.data import DataLoader, TensorDataset
from .. import DataWrapper
import numpy as np
import torch


class STGCNDataWrapper(DataWrapper):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--batch_size", type=int, default=30)
        parser.add_argument("--n_his", type=int, default=20)
        parser.add_argument("--n_pred", type=int, default=1)
        parser.add_argument("--train_prop", type=int, default=0.8)
        parser.add_argument("--val_prop", type=int, default=0.1)
        parser.add_argument("--test_prop", type=int, default=0.1)
        parser.add_argument("--pred_length", type=int, default=288)
        # fmt: on

    def __init__(self, dataset, **args):
        super(STGCNDataWrapper, self).__init__(dataset)
        self.dataset = dataset
        print(args)
        print(args['train_prop'])
        self.train_prop = args['train_prop']
        self.val_prop = args['val_prop']
        self.test_prop = args['test_prop']
        self.pred_length = args['pred_length']
        self.n_his = args['n_his']
        self.n_pred = args['n_pred']
        self.batch_size = args['batch_size']
        self.scaler = dataset.data.scaler


    def train_wrapper(self):
        train_data = self.dataLoad()[0]
        return DataLoader(train_data, self.batch_size, shuffle=True)

    def val_wrapper(self):
        val_data = self.dataLoad()[1]
        return DataLoader(val_data, self.batch_size, shuffle=False)

    def test_wrapper(self):
        # test_data = self.dataLoad()[2]
        # return DataLoader(test_data, self.batch_size, shuffle=False)

        pred_data = self.dataLoad()[3]
        return DataLoader(pred_data, self.pred_length + self.n_his + self.n_pred + 1, shuffle=False)

    def predict_wrapper(self):
        pred_data = self.dataLoad()[3]
        return DataLoader(pred_data, self.pred_length + self.n_his + self.n_pred + 1, shuffle=False)

    def data_transform(self, data, n_his, n_pred, device):
        # data = slice of V matrix
        # n_his = number of historical speed observations to consider
        # n_pred = number of time steps in the future to predict
        num_nodes = data.shape[1]
        num_obs = len(data) - n_his - n_pred
        x = np.zeros([num_obs, n_his, num_nodes, 1]) 
        y = np.zeros([num_obs, num_nodes])
        obs_idx = 0
        for i in range(num_obs):
            head = i
            tail = i + n_his
            x[obs_idx, :, :, :] = data[head: tail].reshape(n_his, num_nodes, 1)
            y[obs_idx] = data[tail + n_pred - 1]
            obs_idx += 1
        return torch.Tensor(x).to(device), torch.Tensor(y).to(device)


    def dataLoad(self):
        V = self.dataset.data.V
        len_train = round(self.dataset.data.num_samples * self.train_prop)
        len_val = round(self.dataset.data.num_samples * self.val_prop)
        len_test = round(self.dataset.data.num_samples * self.test_prop)
        train = V[: len_train]
        val = V[len_train: len_train + len_val]
        test = V[len_train + len_val: len_train + len_val + len_test] 

        pred_set = V[-(self.n_his + self.n_pred+ 1):]

        train = np.nan_to_num(self.scaler.fit_transform(train))
        val = np.nan_to_num(self.scaler.transform(val))
        test = np.nan_to_num(self.scaler.transform(test))
        pred_set = np.nan_to_num(self.scaler.transform(pred_set))

        x_train, y_train = self.data_transform(train, self.n_his, self.n_pred, self.dataset.data.device)
        x_val, y_val = self.data_transform(val, self.n_his, self.n_pred, self.dataset.data.device)
        x_test, y_test = self.data_transform(test, self.n_his, self.n_pred, self.dataset.data.device)

        x_pred, y_pred = self.data_transform(pred_set, self.n_his, self.n_pred, self.dataset.data.device)

        # create torch data iterables for training
        train_data = TensorDataset(x_train, y_train)
        # train_iter = DataLoader(train_data, self.batch_size, shuffle=True)
        val_data = TensorDataset(x_val, y_val)
        # val_iter = DataLoader(val_data, self.batch_size, shuffle=False)
        test_data = TensorDataset(x_test, y_test)
        # test_iter = DataLoader(test_data, self.batch_size, shuffle=False)

        pred_data = TensorDataset(x_pred, y_pred)
        # pred_iter = DataLoader(pred_data, self.n_his + self.n_pred + 1, shuffle=False)

        return [train_data, val_data, test_data, pred_data]


    def get_pre_timestamp(self):
        # len_train = round(self.dataset.data.num_samples * self.train_prop)
        # len_val = round(self.dataset.data.num_samples * self.val_prop)
        # pred_set_timestamp = self.dataset.data.timestamp[len_train + len_val: len_train + len_val + self.pred_length][-self.pred_length:]
        pred_set_timestamp = self.dataset.data.timestamp[-self.pred_length:]
        return pred_set_timestamp














