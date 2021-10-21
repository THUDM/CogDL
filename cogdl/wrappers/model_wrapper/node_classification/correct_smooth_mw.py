import copy

import torch
import numpy as np

from cogdl.wrappers.data_wrapper import DataWrapper

from .node_classification_mw import NodeClfModelWrapper


class CorrectSmoothModelWrapper(NodeClfModelWrapper):
    @staticmethod
    def add_args(parser):
        pass

    def __init__(self, model, optimizer_cfg):
        super(CorrectSmoothModelWrapper, self).__init__(model, optimizer_cfg)
        self.model = model
        self.optimizer_cfg = optimizer_cfg

    def val_step(self, subgraph):
        graph = subgraph
        pred = self.model(graph)
        pred = self.model.postprocess(graph, pred)
        y = graph.y
        val_mask = graph.val_mask
        loss = self.default_loss_fn(pred[val_mask], y[val_mask])

        metric = self.evaluate(pred[val_mask], graph.y[val_mask], metric="auto")

        self.note("val_loss", loss.item())
        self.note("val_metric", metric)

    def test_step(self, batch):
        graph = batch
        pred = self.model(graph)
        pred = self.model.postprocess(graph, pred)
        test_mask = batch.test_mask
        loss = self.default_loss_fn(pred[test_mask], batch.y[test_mask])

        metric = self.evaluate(pred[test_mask], batch.y[test_mask], metric="auto")

        self.note("test_loss", loss.item())
        self.note("test_metric", metric)
