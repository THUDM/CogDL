import torch
from .. import ModelWrapper
import numpy as np
import warnings


class STGCNModelWrapper(ModelWrapper):
    def __init__(self, model, optimizer_cfg, **args):
        super(STGCNModelWrapper, self).__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.edge_index = torch.stack(args['edge_index'])
        self.edge_weight = args['edge_weight']
        self.scaler = args['scaler']
        self.node_ids = args['node_ids']
        self.pred_timestamp = args['pred_timestamp']


    def train_step(self, batch):
        batch_x, y = batch
        pred = self.model(batch_x, self.edge_index, self.edge_weight)
        loss = self.default_loss_fn(pred.view(len(batch_x), -1), y)
        return loss


    def val_step(self, batch):
        batch_x, y = batch
        pred = self.model(batch_x, self.edge_index, self.edge_weight)


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
        batch_x, y = batch
        pre_len = batch_x.shape[1]


        pred = self.model(batch_x, self.edge_index, self.edge_weight)

        _y = self.scaler.inverse_transform(y.cpu()).reshape(-1)
        _pred = self.scaler.inverse_transform(pred.view(len(batch_x), -1).cpu().numpy()).reshape(-1)
        mae = self.evaluate(torch.from_numpy(_pred), torch.from_numpy(_y))

        self.note("test_loss", self.default_loss_fn(pred.view(len(batch_x), -1), y))
        self.note("test_metric", mae)

        #### get predictions for last day
        self.model.eval()
        import pandas as pd
        import datetime

        pre_pd = pd.DataFrame()
        timestamp = self.pred_timestamp.tolist()
        all_pre_y = torch.zeros(288,288)
        with torch.no_grad():
            for i in range(len(self.node_ids)):
                timestamp[i] = datetime.datetime.strptime(timestamp[i], '%m/%d/%Y %H:%M:%S') + datetime.timedelta(days=1)
                for j in range(pre_len-1):
                    batch_x[0][j] = batch_x[0][j+1]
                batch_x[0][pre_len-1] = y.view(-1,1)
                pred = self.model(batch_x, self.edge_index, self.edge_weight)
                y = pred.view(1,-1)
                all_pre_y[i] = y

        pre_y = self.scaler.inverse_transform(all_pre_y.cpu().numpy()).reshape(-1, len(self.node_ids))
        pre_pd['timestamp'] = timestamp
        for i in range(len(self.node_ids)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pre_pd[self.node_ids[i]] = list(pre_y[:,i])
        pre_pd.to_csv('./data/pems-stgcn/stgcn_prediction.csv',index=False)

    # def predict_step(self, batch):
    #     batch_x, y = batch
    #     pred = self.model(batch_x, self.edge_index, self.edge_weight)
    #
    #
    #    _y = self.scaler.inverse_transform(y.cpu()).reshape(-1)
    #    _pred = self.scaler.inverse_transform(pred.view(len(batch_x), -1).cpu().numpy()).reshape(-1)


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
                pred = self.model(batch_x, batch_y_emb)
                preds.append(pred.to("cpu"))
        probs = torch.cat(preds, dim=0)
        return probs


    def setup_optimizer(self):
        cfg = self.optimizer_cfg
        return torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


    def set_early_stopping(self):
        return "val_metric", "<"


def evaluate_model(model, loss, x, y, edge_index, edge_weight):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        y_pred = model(x, edge_index, edge_weight).view(len(x), -1)
        tmp_loss = loss(y_pred, y)
        l_sum += tmp_loss.item() * y.shape[0]
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
