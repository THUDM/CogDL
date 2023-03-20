import math
from cogdl import function as BF
from cogdl.backend import BACKEND

if BACKEND == "jittor":
    import jittor
    from jittor import nn, Module
    from jittor import nn as F
elif BACKEND == "torch":
    import torch
    from torch.nn import Module
    import torch.nn as nn
    import torch.nn.functional as F


class MemoryMoCo(Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize  # None
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer("params", BF.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer("memory", BF.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))

    if BACKEND == "jittor":

        def execute(self, q, k):
            batchSize = q.shape[0]
            k = k.detach()

            Z = self.params[0].item()

            # pos logit
            l_pos = jittor.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
            l_pos = l_pos.view(batchSize, 1)
            # neg logit
            queue = self.memory.clone()
            l_neg = jittor.matmul(queue.detach(), q.transpose(1, 0))
            l_neg = l_neg.transpose(0, 1)

            out = jittor.concat((l_pos, l_neg), dim=1)

            if self.use_softmax:
                out = jittor.divide(out, self.T)
                out = BF.squeeze(out).contiguous()
            else:
                out = jittor.exp(jittor.divide(out, self.T))
                if Z < 0:
                    self.params[0] = out.mean() * self.outputSize
                    Z = self.params[0].clone().detach().item()
                    print("normalization constant Z is set to {:.1f}".format(Z))
                # compute the out
                out = BF.squeeze(jittor.divide(out, Z)).contiguous()
                # # update memory
            with jittor.no_grad():
                out_ids = jittor.arange(batchSize, device=out.device)
                out_ids += self.index
                out_ids = jittor.mod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, k)
                self.index = (self.index + batchSize) % self.queueSize

            return out

    elif BACKEND == "torch":

        def forward(self, q, k):
            batchSize = q.shape[0]
            k = k.detach()

            Z = self.params[0].item()

            # pos logit
            l_pos = torch.bmm(q.view(batchSize, 1, -1), k.view(batchSize, -1, 1))
            l_pos = l_pos.view(batchSize, 1)
            # neg logit
            queue = self.memory.clone()
            l_neg = torch.mm(queue.detach(), q.transpose(1, 0))
            l_neg = l_neg.transpose(0, 1)

            out = torch.cat((l_pos, l_neg), dim=1)

            if self.use_softmax:
                out = torch.div(out, self.T)
                out = out.squeeze().contiguous()
            else:
                out = torch.exp(torch.div(out, self.T))
                if Z < 0:
                    self.params[0] = out.mean() * self.outputSize
                    Z = self.params[0].clone().detach().item()
                    print("normalization constant Z is set to {:.1f}".format(Z))
                # compute the out
                out = torch.div(out, Z).squeeze().contiguous()

            # # update memory
            with torch.no_grad():
                out_ids = torch.arange(batchSize, device=out.device)
                out_ids += self.index
                out_ids = torch.fmod(out_ids, self.queueSize)
                out_ids = out_ids.long()
                self.memory.index_copy_(0, out_ids, k)
                self.index = (self.index + batchSize) % self.queueSize

            return out


class NCESoftmaxLoss(Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    if BACKEND == "jittor":

        def execute(self, x):
            bsz = x.shape[0]
            x = BF.squeeze(x)
            label = BF.zeros([bsz]).long()
            loss = self.criterion(x, label)
            return loss

    elif BACKEND == "torch":

        def forward(self, x):
            bsz = x.shape[0]
            x = x.squeeze()
            label = torch.zeros([bsz], device=x.device).long()
            loss = self.criterion(x, label)
            return loss


def moment_update(model, model_ema, m):
    """model_ema = m * model_ema + (1 - m) model"""
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        # p2.data.mul_(m).add_(1 - m, p1.detach().data)
        p2.data.mul_(m).add_(p1.detach().data * (1 - m))
