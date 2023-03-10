import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from .. import BaseModel
from cogdl.utils import spmm


class GraphConvolutionBS(Module):
    """
    GCN Layer with BN, Self-loop and Res connection.
    """

    def __init__(
        self, in_features, out_features, activation=lambda x: x, withbn=True, withloop=True, bias=True, res=False,
    ):
        """
        Initial function.
        :param in_features: the input feature dimension.
        :param out_features: the output feature dimension.
        :param activation: the activation function.
        :param withbn: using batch normalization.
        :param withloop: using self feature modeling.
        :param bias: enable bias.
        :param res: enable res connections.
        """
        super(GraphConvolutionBS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.res = res

        # Parameter setting.
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.self_weight = Parameter(torch.FloatTensor(in_features, out_features)) if withloop else None

        self.bn = torch.nn.BatchNorm1d(out_features) if withbn else None
        self.bias = Parameter(torch.FloatTensor(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_weight is not None:
            stdv = 1.0 / math.sqrt(self.self_weight.size(1))
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, x):
        support = torch.mm(x, self.weight)
        output = spmm(graph, support)

        # Self-loop
        output = output + torch.mm(x, self.self_weight) if self.self_weight is not None else output

        output = output + self.bias if self.bias is not None else output
        # BN
        output = self.bn(output) if self.bn is not None else output
        # Res
        return self.sigma(output) + input if self.res else self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class GraphBaseBlock(Module):
    """
    The base block for Multi-layer GCN / ResGCN / Dense GCN
    """

    def __init__(
        self,
        in_features,
        out_features,
        nbaselayer,
        withbn=True,
        withloop=True,
        activation=F.relu,
        dropout=True,
        aggrmethod="concat",
        dense=False,
    ):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(GraphBaseBlock, self).__init__()
        self.in_features = in_features
        self.hiddendim = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat" and dense is False:
            self.out_features = in_features + out_features
        elif self.aggrmethod == "concat" and dense is True:
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in add model.")
            self.out_features = out_features
        elif self.aggrmethod == "nores":
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat','add' and 'nores'.")

    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer = GraphConvolutionBS(
                    self.in_features, self.hiddendim, self.activation, self.withbn, self.withloop,
                )
            else:
                layer = GraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop,)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, graph, x):
        h = x
        denseout = None
        # Here out is the result in all levels.
        for gc in self.hiddenlayers:
            denseout = self._doconcat(denseout, h)
            h = gc(graph, h)
            h = F.dropout(h, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(h, x)

        return self._doconcat(h, denseout) if denseout is not None else h

    def get_outdim(self):
        return self.out_features

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (
            self.__class__.__name__,
            self.aggrmethod,
            self.in_features,
            self.hiddendim,
            self.nhiddenlayer,
            self.out_features,
        )


class MultiLayerGCNBlock(Module):
    """
    Muti-Layer GCN with same hidden dimension.
    """

    def __init__(
        self,
        in_features,
        out_features,
        nbaselayer,
        withbn=True,
        withloop=True,
        activation=F.relu,
        dropout=True,
        aggrmethod=None,
        dense=None,
    ):
        """
        The multiple layer GCN block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(MultiLayerGCNBlock, self).__init__()
        self.model = GraphBaseBlock(
            in_features=in_features,
            out_features=out_features,
            nbaselayer=nbaselayer,
            withbn=withbn,
            withloop=withloop,
            activation=activation,
            dropout=dropout,
            dense=False,
            aggrmethod="nores",
        )

    def forward(self, graph, x):
        return self.model.forward(graph, x)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (
            self.__class__.__name__,
            self.aggrmethod,
            self.model.in_features,
            self.model.hiddendim,
            self.model.nhiddenlayer,
            self.model.out_features,
        )


class ResGCNBlock(Module):
    """
    The multiple layer GCN with residual connection block.
    """

    def __init__(
        self,
        in_features,
        out_features,
        nbaselayer,
        withbn=True,
        withloop=True,
        activation=F.relu,
        dropout=True,
        aggrmethod=None,
        dense=None,
    ):
        """
        The multiple layer GCN with residual connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: not applied.
        :param dense: not applied.
        """
        super(ResGCNBlock, self).__init__()
        self.model = GraphBaseBlock(
            in_features=in_features,
            out_features=out_features,
            nbaselayer=nbaselayer,
            withbn=withbn,
            withloop=withloop,
            activation=activation,
            dropout=dropout,
            dense=False,
            aggrmethod="add",
        )

    def forward(self, graph, x):
        return self.model.forward(graph, x)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (
            self.__class__.__name__,
            self.aggrmethod,
            self.model.in_features,
            self.model.hiddendim,
            self.model.nhiddenlayer,
            self.model.out_features,
        )


class DenseGCNBlock(Module):
    """
    The multiple layer GCN with dense connection block.
    """

    def __init__(
        self,
        in_features,
        out_features,
        nbaselayer,
        withbn=True,
        withloop=True,
        activation=F.relu,
        dropout=True,
        aggrmethod="concat",
        dense=True,
    ):
        """
        The multiple layer GCN with dense connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for the output. For denseblock, default is "concat".
        :param dense: default is True, cannot be changed.
        """
        super(DenseGCNBlock, self).__init__()
        self.model = GraphBaseBlock(
            in_features=in_features,
            out_features=out_features,
            nbaselayer=nbaselayer,
            withbn=withbn,
            withloop=withloop,
            activation=activation,
            dropout=dropout,
            dense=True,
            aggrmethod=aggrmethod,
        )

    def forward(self, graph, x):
        return self.model.forward(graph, x)

    def get_outdim(self):
        return self.model.get_outdim()

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (
            self.__class__.__name__,
            self.aggrmethod,
            self.model.in_features,
            self.model.hiddendim,
            self.model.nhiddenlayer,
            self.model.out_features,
        )


class InceptionGCNBlock(Module):
    """
    The multiple layer GCN with inception connection block.
    """

    def __init__(
        self,
        in_features,
        out_features,
        nbaselayer,
        withbn=True,
        withloop=True,
        activation=F.relu,
        dropout=True,
        aggrmethod="concat",
        dense=False,
    ):
        """
        The multiple layer GCN with inception connection block.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: not applied. The default is False, cannot be changed.
        """
        super(InceptionGCNBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hiddendim = out_features
        self.nbaselayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.midlayers = nn.ModuleList()
        self.__makehidden()

        if self.aggrmethod == "concat":
            self.out_features = in_features + out_features * nbaselayer
        elif self.aggrmethod == "add":
            if in_features != self.hiddendim:
                raise RuntimeError("The dimension of in_features and hiddendim should be matched in 'add' model.")
            self.out_features = out_features
        else:
            raise NotImplementedError("The aggregation method only support 'concat', 'add'.")

    def __makehidden(self):
        # for j in xrange(self.nhiddenlayer):
        for j in range(self.nbaselayer):
            reslayer = nn.ModuleList()
            # for i in xrange(j + 1):
            for i in range(j + 1):
                if i == 0:
                    layer = GraphConvolutionBS(
                        self.in_features, self.hiddendim, self.activation, self.withbn, self.withloop,
                    )
                else:
                    layer = GraphConvolutionBS(
                        self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop,
                    )
                reslayer.append(layer)
            self.midlayers.append(reslayer)

    def forward(self, graph, x):
        for reslayer in self.midlayers:
            subx = x
            for gc in reslayer:
                subx = gc(graph, x)
                subx = F.dropout(subx, self.dropout, training=self.training)
            x = self._doconcat(x, subx)
        return x

    def get_outdim(self):
        return self.out_features

    def _doconcat(self, x, subx):
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx

    def __repr__(self):
        return "%s %s (%d - [%d:%d] > %d)" % (
            self.__class__.__name__,
            self.aggrmethod,
            self.in_features,
            self.hiddendim,
            self.nbaselayer,
            self.out_features,
        )


class Dense(Module):
    """
    Simple Dense layer, Do not consider adj.
    """

    def __init__(self, in_features, out_features, activation=lambda x: x, bias=True, res=False):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = activation
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.res = res
        self.bn = nn.BatchNorm1d(out_features)
        self.bias = Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, graph, x):
        output = torch.mm(x, self.weight)
        output = output + self.bias if self.bias is not None else output
        output = self.bn(output)
        return self.sigma(output)

    def __repr__(self):
        return self.__class__.__name__ + " (" + str(self.in_features) + " -> " + str(self.out_features) + ")"


class DropEdge_GCN(BaseModel):
    """
     DropEdge: Towards Deep Graph Convolutional Networks on Node Classification
     Applying DropEdge to GCN @ https://arxiv.org/pdf/1907.10903.pdf

    The model for the single kind of deepgcn blocks.
    The model architecture likes:
    inputlayer(nfeat)--block(nbaselayer, nhid)--...--outputlayer(nclass)--softmax(nclass)
                        |------  nhidlayer  ----|
    The total layer is nhidlayer*nbaselayer + 2.
    All options are configurable.

     Args:
         Initial function.
         :param nfeat: the input feature dimension.
         :param nhid:  the hidden feature dimension.
         :param nclass: the output feature dimension.
         :param nhidlayer: the number of hidden blocks.
         :param dropout:  the dropout ratio.
         :param baseblock: the baseblock type, can be "mutigcn", "resgcn", "densegcn" and "inceptiongcn".
         :param inputlayer: the input layer type, can be "gcn", "dense", "none".
         :param outputlayer: the input layer type, can be "gcn", "dense".
         :param nbaselayer: the number of layers in one hidden block.
         :param activation: the activation function, default is ReLu.
         :param withbn: using batch normalization in graph convolution.
         :param withloop: using self feature modeling in graph convolution.
         :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                            is "add", for others the default is "concat".
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num_features", type=int)

        parser.add_argument("--num_classes", type=int)

        parser.add_argument('--baseblock', default='mutigcn',
                            help="Choose the model to be trained.( mutigcn, resgcn, densegcn, inceptiongcn)")
        parser.add_argument('--inputlayer', default='gcn',
                            help="The input layer of the model.")
        parser.add_argument('--outputlayer', default='gcn',
                            help="The output layer of the model.")
        parser.add_argument('--hidden-size', type=int, default=64,
                            help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (1 - keep probability).')
        parser.add_argument('--withbn', action='store_true', default=False,
                            help='Enable Bath Norm GCN')
        parser.add_argument('--withloop', action="store_true", default=False,
                            help="Enable loop layer GCN")
        parser.add_argument('--nhiddenlayer', type=int, default=1,
                            help='The number of hidden layers.')

        parser.add_argument("--nbaseblocklayer", type=int, default=0,
                            help="The number of layers in each baseblock")
        parser.add_argument("--aggrmethod", default="default",
                            help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inceptiongcn")
        parser.add_argument("--task_type", default="full", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

        parser.add_argument("--activation", default=F.relu, help="activiation function")

        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.hidden_size,
            args.num_classes,
            args.nhiddenlayer,
            args.dropout,
            args.baseblock,
            args.inputlayer,
            args.outputlayer,
            args.nbaseblocklayer,
            args.activation,
            args.withbn,
            args.withloop,
            args.aggrmethod,
        )

    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nhidlayer,
        dropout,
        baseblock,
        inputlayer,
        outputlayer,
        nbaselayer,
        activation,
        withbn,
        withloop,
        aggrmethod,
    ):
        super(DropEdge_GCN, self).__init__()

        self.dropout = dropout

        if baseblock == "resgcn":
            self.BASEBLOCK = ResGCNBlock
        elif baseblock == "densegcn":
            self.BASEBLOCK = DenseGCNBlock
        elif baseblock == "mutigcn":
            self.BASEBLOCK = MultiLayerGCNBlock
        elif baseblock == "inceptiongcn":
            self.BASEBLOCK = InceptionGCNBlock
        else:
            raise NotImplementedError("Current baseblock %s is not supported." % (baseblock))
        if inputlayer == "gcn":
            # input gc
            self.ingc = GraphConvolutionBS(nfeat, nhid, activation, withbn, withloop)
            baseblockinput = nhid
        elif inputlayer == "none":
            self.ingc = lambda x: x
            baseblockinput = nfeat
        else:
            self.ingc = Dense(nfeat, nhid, activation)
            baseblockinput = nhid

        outactivation = lambda x: x  # noqa E731
        if outputlayer == "gcn":
            self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)
        # elif outputlayer ==  "none": #here can not be none
        #    self.outgc = lambda x: x
        else:
            self.outgc = Dense(nhid, nclass, activation)

        # hidden layer
        self.midlayer = nn.ModuleList()
        # Dense is not supported now.
        # for i in xrange(nhidlayer):
        for i in range(nhidlayer):
            gcb = self.BASEBLOCK(
                in_features=baseblockinput,
                out_features=nhid,
                nbaselayer=nbaselayer,
                withbn=withbn,
                withloop=withloop,
                activation=activation,
                dropout=dropout,
                dense=False,
                aggrmethod=aggrmethod,
            )
            self.midlayer.append(gcb)
            baseblockinput = gcb.get_outdim()
        # output gc
        outactivation = lambda x: x  # noqa E731 we donot need nonlinear activation here.
        self.outgc = GraphConvolutionBS(baseblockinput, nclass, outactivation, withbn, withloop)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, graph):
        x = graph.x

        x = self.ingc(graph, x)
        x = F.dropout(x, self.dropout, training=self.training)

        # mid block connections
        # for i in xrange(len(self.midlayer)):
        for i in range(len(self.midlayer)):
            midgc = self.midlayer[i]
            x = midgc(graph, x)
        # output, no relu and dropput here.
        x = self.outgc(graph, x)
        x = F.log_softmax(x, dim=1)
        return x

    def predict(self, data):
        return self.forward(data)
