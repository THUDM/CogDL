import math

import torch
import torch.nn as nn


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, inputSize, outputSize, K, T=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.outputSize = outputSize  # None
        self.inputSize = inputSize
        self.queueSize = K
        self.T = T
        self.index = 0
        self.use_softmax = use_softmax

        self.register_buffer("params", torch.tensor([-1]))
        stdv = 1.0 / math.sqrt(inputSize / 3)
        self.register_buffer("memory", torch.rand(self.queueSize, inputSize).mul_(2 * stdv).add_(-stdv))
        print("using queue shape: ({},{})".format(self.queueSize, inputSize))

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


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

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
