import torch
import torch.nn as nn
import torch.nn.functional as F
from cogdl.modules.conv import  meanaggr

from . import register_model, BaseModel
import random
import numpy as np
@register_model("fastgcn")
class FastGCN(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--num-features", type=int)
        parser.add_argument("--num-classes", type=int)
        parser.add_argument("--hidden-size", type=int, nargs='+',default=[128])
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--sample-size",type=int,nargs='+',default=[512,256,256])
        parser.add_argument("--dropout", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.num_layers,
            args.sample_size,
            args.dropout,
        )
    # edge index based sampler
    #@profile
    def construct_adjlist(self,edge_index):
        # print(edge_index)
        if self.adjlist=={}:
            
            edge_index=edge_index.t().cpu().tolist()
            for i in edge_index:
                if not(i[0] in self.adjlist):
                    self.adjlist[i[0]]=[i[1]]
                else:
                    self.adjlist[i[0]].append(i[1])
                        #print(self.adjlist)
        return 0
    
    #@profile
    def generate_index(self,sample1,sample2):
        edgelist=[]
        values=[]
        iddict1={}
        for i in range(len(sample1)):
            iddict1[sample1[i]]=i

        for i in range(len(sample2)):
            case=self.adjlist[sample2[i]]
            for adj in case:
                if adj in iddict1:
                    edgelist.append([i,iddict1[adj]])
                    values.append(1)
        edgetensor=torch.LongTensor(edgelist)
        valuetensor=torch.FloatTensor(values)
        #print(edgetensor,valuetensor,len(sample2),len(sample1))
        t=torch.sparse.FloatTensor(edgetensor.t(),valuetensor,torch.Size([len(sample2),len(sample1)])).cuda()
        return t
                                   
    #@profile
    def sample_one_layer(self, init_index,sample_size):
        alllist=[]
        for i in init_index:
            alllist.extend(self.adjlist[i])
        alllist=list(np.unique(alllist))
        
        if sample_size>len(alllist):
            sample_size=len(alllist)
                #print(init_index,alllist,sample_size)
        alllist=random.sample(alllist,sample_size)
        return alllist
    
    def __init__(self, num_features, num_classes, hidden_size,num_layers,sample_size, dropout):
        super(FastGCN, self).__init__()
        self.adjlist={}
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sample_size=sample_size
        self.dropout = dropout
        shapes = [num_features] + hidden_size+ [num_classes]
        #print(shapes)
        self.convs = nn.ModuleList(
            [
                meanaggr(shapes[layer], shapes[layer + 1], cached=True)
                for layer in range(num_layers)
            ]
        )
    
    #@profile
    def forward(self, x,train_index):
        sampled=[[]]*(self.num_layers+1)
        sampled[self.num_layers]=train_index
        # print("train",train_index)
        for i in range(self.num_layers-1,-1,-1):
            sampled[i]=self.sample_one_layer(sampled[i+1],self.sample_size[i])
        #construct_tensor
        w=torch.LongTensor(sampled[0]).cuda()
        x=torch.index_select(x,0,w)
        
        for i in range(self.num_layers):
            # print(i,len(sampled[i]),len(sampled[i+1]))
            edge_index_sp=self.generate_index(sampled[i],sampled[i+1])
            x = self.convs[i](x, edge_index_sp,sampled[i+1])
            if i!=self.num_layers-1:
                x=F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
    
        return F.log_softmax(x, dim=1)
