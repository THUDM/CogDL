from functools import partial

import torch
import torch.nn.functional as F

from .. import BaseModel
from .mlp import MLP
from cogdl.data import Graph
from cogdl.utils import spmm


def autoscale_post(x, lower, upper):
    return torch.clamp(x, lower, upper)


def fixed_post(x, y, nid):
    x[nid] = y[nid]
    return x


def pre_residual_correlation(preds, labels, split_idx):
    labels[labels.isnan()] = 0
    labels = labels.long()
    nclass = labels.max().item() + 1
    nnode = preds.shape[0]
    err = torch.zeros((nnode, nclass), device=preds.device)
    err[split_idx] = F.one_hot(labels[split_idx], nclass).float().squeeze(1) - preds[split_idx]
    return err


def pre_outcome_correlation(preds, labels, label_nid):
    """Generates the initial labels used for outcome correlation"""
    c = labels.max() + 1
    y = preds.clone()
    if len(label_nid) > 0:
        y[label_nid] = F.one_hot(labels[label_nid], c).float().squeeze(1)
    return y


def outcome_correlation(g, labels, alpha, nprop, post_step, alpha_term=True):
    result = labels.clone()
    for _ in range(nprop):
        result = alpha * spmm(g, result)
        if alpha_term:
            result += (1 - alpha) * labels
        else:
            result += labels
        result = post_step(result)
    return result


def correlation_autoscale(preds, y, resid, residual_nid, scale=1.0):
    orig_diff = y[residual_nid].abs().sum() / residual_nid.shape[0]
    resid_scale = orig_diff / resid.abs().sum(dim=1, keepdim=True)
    resid_scale[resid_scale.isinf()] = 1.0
    cur_idxs = resid_scale > 1000
    resid_scale[cur_idxs] = 1.0
    res_result = preds + resid_scale * resid
    res_result[res_result.isnan()] = preds[res_result.isnan()]
    return res_result


def correlation_fixed(preds, y, resid, residual_nid, scale=1.0):
    return preds + scale * resid


def diffusion(g, x, nhtop, p=1, alpha=0.5):
    x = x ** p
    for _ in range(nhtop):
        x = (1 - alpha) * x + alpha * spmm(g, x)
        x = x ** p
    return x


class CorrectSmooth(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--correct-alpha", type=float, default=1.0)
        parser.add_argument("--smooth-alpha", type=float, default=0.8)
        parser.add_argument("--num-correct-prop", type=int, default=50)
        parser.add_argument("--num-smooth-prop", type=int, default=50)
        parser.add_argument("--autoscale", action="store_true")
        parser.add_argument("--correct-norm", type=str, default="sym")
        parser.add_argument("--smooth-norm", type=str, default="row")
        parser.add_argument("--scale", type=float, default=1.0)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.correct_alpha,
            args.smooth_alpha,
            args.num_correct_prop,
            args.num_smooth_prop,
            args.autoscale,
            args.correct_norm,
            args.smooth_norm,
            args.scale,
        )

    def __init__(
        self,
        correct_alpha,
        smooth_alpha,
        num_correct_prop,
        num_smooth_prop,
        autoscale=False,
        correct_norm="row",
        smooth_norm="col",
        scale=1.0,
    ):
        super(CorrectSmooth, self).__init__()
        self.op_dict = {
            "correct_g": correct_norm,
            "smooth_g": smooth_norm,
            "num_correct_prop": num_correct_prop,
            "num_smooth_prop": num_smooth_prop,
            "correct_alpha": correct_alpha,
            "smooth_alpha": smooth_alpha,
            "autoscale": autoscale,
            "scale": scale,
        }

    def __call__(self, graph, x, train_only=True):
        g1 = graph
        g2 = Graph(edge_index=g1.edge_index)

        g1.normalize(self.op_dict["correct_g"])
        g2.normalize(self.op_dict["smooth_g"])

        train_nid, valid_nid, _ = g1.train_nid, g1.val_nid, g1.test_nid
        y = g1.y

        if train_only:
            label_nid = train_nid
            residual_nid = train_nid
        else:
            label_nid = torch.cat((train_nid, valid_nid))
            residual_nid = train_nid

        # Correct
        y = pre_residual_correlation(x, y, residual_nid)

        if self.op_dict["autoscale"]:
            post_func = partial(autoscale_post, lower=-1.0, upper=1.0)
            scale_func = correlation_autoscale
        else:
            post_func = partial(fixed_post, y=y, nid=residual_nid)
            scale_func = correlation_fixed

        resid = outcome_correlation(
            g1, y, self.op_dict["correct_alpha"], nprop=self.op_dict["num_correct_prop"], post_step=post_func
        )
        res_result = scale_func(x, y, resid, residual_nid, self.op_dict["scale"])

        # Smooth
        y = pre_outcome_correlation(res_result, g1.y, label_nid)
        result = outcome_correlation(
            g2,
            y,
            self.op_dict["smooth_alpha"],
            nprop=self.op_dict["num_smooth_prop"],
            post_step=partial(autoscale_post, lower=0, upper=1),
        )
        return result


class CorrectSmoothMLP(BaseModel):
    @staticmethod
    def add_args(parser):
        CorrectSmooth.add_args(parser)
        MLP.add_args(parser)
        parser.add_argument("--use-embeddings", action="store_true")

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(CorrectSmoothMLP, self).__init__()
        if args.use_embeddings:
            args.num_features = args.num_features * 2
        args.act_first = True if args.dataset == "ogbn-products" else False
        self.use_embeddings = args.use_embeddings
        self.mlp = MLP.build_model_from_args(args)
        self.c_s = CorrectSmooth.build_model_from_args(args)
        self.rescale_feats = args.rescale_feats if hasattr(args, "rescale_feats") else args.dataset == "ogbn-arxiv"
        self.cache_x = None

    def forward(self, graph):
        if self.cache_x is not None:
            x = self.cache_x
        elif self.use_embeddings:
            _x = graph.x.contiguous()
            _x = diffusion(graph, _x, nhtop=10)
            x = torch.cat([graph.x, _x], dim=1)
            if self.rescale_feats:
                x = (x - x.mean(0)) / x.std(0)
            self.cache_x = x
        else:
            x = graph.x
        out = self.mlp(x)
        return out

    def predict(self, data):
        out = self.forward(data)
        return out

    def postprocess(self, data, out):
        if len(data.y.shape) == 1:
            out = F.softmax(out, dim=-1)
        # else:
        # out = torch.sigmoid(out)
        out = self.c_s(data, out)
        return out
