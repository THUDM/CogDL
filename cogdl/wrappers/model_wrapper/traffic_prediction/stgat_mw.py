import torch
from .. import ModelWrapper
import numpy as np


class STGATModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg, **args):
        super(STGATModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.edge_index = torch.stack(args['edge_index'])
        self.edge_weight = args['edge_weight']
        self.scaler = args['scaler']
        self.node_ids = args['node_ids']
        self.pred_timestamp = args['pred_timestamp']


    def train_step(self, batch):
        batch_x, y = batch
        batch_size, num_feature, _, _ = batch_x.size()
        xxx = batch_x.view(-1, num_feature)
        pred = self.model(xxx, self.edge_index, self.edge_weight,batch_size , num_feature)
        loss = self.default_loss_fn(pred.view(len(batch_x), -1), y)
        return loss


    def val_step(self, batch):
        batch_x, y = batch
        batch_size, num_feature, _, _ = batch_x.size()
        xxx = batch_x.view(-1, num_feature)
        pred = self.model(xxx, self.edge_index, self.edge_weight,batch_size,num_feature)


        _y = self.scaler.inverse_transform(y.cpu()).reshape(-1)
        _pred = self.scaler.inverse_transform(pred.view(len(batch_x), -1).cpu().numpy()).reshape(-1)
        mae = self.evaluate(torch.from_numpy(_pred), torch.from_numpy(_y))

        # mae, mape, mse = evaluate_metric(self.model, self.scaler, batch_x, y, self.edge_index, self.edge_weight)

        self.note("val_loss", self.default_loss_fn(pred.view(len(batch_x), -1), y))
        self.note("val_metric", mae)

        # self.note("val_mae_metric", mae)
        # self.note("val_mape_metric", mape)
        # self.note("val_mse_metric", mse)



    def test_step(self, batch):
        # 测试步骤我只传了 最后 288 条数据进来 一条一条传进来的
        # batch = 288 + 20 + 1 + 1

        batch_x, y = batch
        pre_len = batch_x.shape[1]
        # pre_len = 20
        #  batch_x    torch.Size([1, 20, 288, 1])
        # y    torch.Size([1, 288])
        batch_size, num_feature, _, _ = batch_x.size()
        xxx = batch_x.view(-1, num_feature)
        pred = self.model(xxx, self.edge_index, self.edge_weight, batch_size, num_feature)
        # pred   torch.Size([1, 1, 1, 288])

        _y = self.scaler.inverse_transform(y.cpu()).reshape(-1)
        _pred = self.scaler.inverse_transform(pred.view(len(batch_x), -1).cpu().numpy()).reshape(-1)
        mae = self.evaluate(torch.from_numpy(_pred), torch.from_numpy(_y))

        # mae, mape, mse = evaluate_metric(self.model, self.scaler, batch_x, y, self.edge_index, self.edge_weight)

        self.note("test_loss", self.default_loss_fn(pred.view(len(batch_x), -1), y))
        self.note("test_metric", mae)
        # self.note("test_mae_metric", mae)
        # self.note("test_mape_metric", mape)
        # self.note("test_mse_metric", mse)

        #### get predictions for last day
        self.model.eval()
        import pandas as pd
        import datetime

        pre_pd = pd.DataFrame()
        # 时间是由 288 条数据给到的
        timestamp = self.pred_timestamp.tolist()
        all_pre_y = torch.zeros(288,288)
        with torch.no_grad():
            # 这一步是 将 289 预测出来后，再放入到 x 用预测出来的289 去 预测 290，直到我们预测出来了 全新的 288 条数据
            for i in range(len(self.node_ids)):
                timestamp[i] = datetime.datetime.strptime(timestamp[i], '%m/%d/%Y %H:%M:%S') + datetime.timedelta(days=1)
                # 288条数据的时间
                # 遍历 288 次
                for j in range(pre_len-1):
                    # 所有数据前移一个单位
                    # 即 20条数据 不断的移动
                    batch_x[0][j] = batch_x[0][j+1]
                # 最有一个单位存 真实值
                batch_x[0][pre_len-1] = y.view(-1,1)
                pred = self.model(batch_x.view(-1, num_feature), self.edge_index, self.edge_weight,batch_size,num_feature)
                y = pred.view(1,-1)
                all_pre_y[i] = y
        # all_pre_y 就是全新的预测数据
        # 再缩放回去得到 pre_y

        pre_y = self.scaler.inverse_transform(all_pre_y.cpu().numpy()).reshape(-1, len(self.node_ids))
        pre_pd['timestamp'] = timestamp
        for i in range(len(self.node_ids)):
            pre_pd[self.node_ids[i]] = list(pre_y[:,i])
        pre_pd.to_csv('./data/pems-stgcn/stgcn_prediction.csv',index=False)



        # # get predictions for last day
        # tru_y, pre_y = get_predictions(self.model, self.scaler, batch_x, y, self.edge_index, self.edge_weight, 288)
        # import pandas as pd
        # pre_pd = pd.DataFrame()
        # tru_pd = pd.DataFrame()
        # tru_pd['timestamp'] = self.pred_timestamp
        # pre_pd['timestamp'] = self.pred_timestamp
        # for i in range(len(self.node_ids)):
        #     tru_pd[self.node_ids[i]] = list(tru_y[:,i])
        #     pre_pd[self.node_ids[i]] = list(pre_y[:,i])
        # tru_pd.to_csv('./data/pems-stgat/stgat_label.csv',index=False)
        # pre_pd.to_csv('./data/pems-stgat/stgat_prediction.csv',index=False)


    def predict_step(self, batch):
        # 想尝试再写一个 有别于 test 的predict 模块，仍在开发中
        batch_x, y = batch
        batch_size, num_feature, _, _ = batch_x.size()
        pred = self.model(batch_x.view(-1, num_feature), self.edge_index, self.edge_weight,batch_size,num_feature)


        _y = self.scaler.inverse_transform(y.cpu()).reshape(-1)
        _pred = self.scaler.inverse_transform(pred.view(len(batch_x), -1).cpu().numpy()).reshape(-1)


        ##### get predictions for last day
        # tru_y, pre_y = get_predictions(self.model, self.scaler, batch_x, y, self.edge_index, self.edge_weight, 288)
        # import pandas as pd
        # pre_pd = pd.DataFrame()
        # tru_pd = pd.DataFrame()
        # tru_pd['timestamp'] = self.pred_timestamp
        # pre_pd['timestamp'] = self.pred_timestamp
        # for i in range(len(self.node_ids)):
        #     tru_pd[self.node_ids[i]] = list(tru_y[:,i])
        #     pre_pd[self.node_ids[i]] = list(pre_y[:,i])
        # tru_pd.to_csv('./data/pems-stgcn/stgcn_label.csv',index=False)
        # pre_pd.to_csv('./data/pems-stgcn/stgcn_prediction.csv',index=False)



    def pre_stage(self, stage, data_w):
        device = next(self.model.parameters()).device
        if stage == 0:
            return None
        self.model.eval()
        preds = []
        eval_loader = data_w.post_stage_wrapper()
        with torch.no_grad():
            for batch in eval_loader:
                batch_x, batch_y_emb, _ = data_w.pre_stage_transform(batch)
                batch_x = batch_x.to(device)
                batch_y_emb = batch_y_emb.to(device) if batch_y_emb is not None else batch_y_emb
                batch_size, num_feature, _, _ = batch_x.size()
                pred = self.model(batch_x.view(-1, num_feature), self.edge_index, self.edge_weight,batch_size,num_feature)
                preds.append(pred.to("cpu"))
        probs = torch.cat(preds, dim=0)
        return probs


    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


    def set_early_stopping(self):
        return "val_metric", "<"



def evaluate_model(model, loss, x, y, edge_index, edge_weight):
    # 计算 所在数据集 的 平均loss
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        y_pred = model(x, edge_index, edge_weight).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, scaler, x, y, edge_index, edge_weight):
    model.eval()
    with torch.no_grad():
        y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
        y_pred = scaler.inverse_transform(model(x, edge_index, edge_weight).view(len(x), -1).cpu().numpy()).reshape(-1)
        d = np.abs(y - y_pred)
        mae = d.tolist()
        mape = (d / y).tolist()
        mse = (d ** 2).tolist()
    MAE = np.array(mae).mean()
    MAPE = np.array(mape).mean()
    RMSE = np.sqrt(np.array(mse).mean())
    return MAE, MAPE, RMSE


def get_predictions(model, scaler, x, y, edge_index, edge_weight, num_nodes):
    model.eval()
    with torch.no_grad():
        y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1, num_nodes)
        _y_pred = scaler.inverse_transform(model(x, edge_index, edge_weight).view(len(x), -1).cpu().numpy()).reshape(-1)
        y_pred = _y_pred.reshape(-1, num_nodes)
        return y, y_pred
