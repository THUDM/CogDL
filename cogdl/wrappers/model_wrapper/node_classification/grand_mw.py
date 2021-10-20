import torch
import torch.nn.functional as F

from cogdl.wrappers.model_wrapper.node_classification.node_classification_mw import NodeClfModelWrapper


class GrandModelWrapper(NodeClfModelWrapper):
    """
    sample : int
        Number of augmentations for consistency loss
    temperature : float
        Temperature to sharpen predictions.
    lmbda : float
         Proportion of consistency loss of unlabelled data
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument("--temperature", type=float, default=0.5)
        parser.add_argument("--lmbda", type=float, default=0.5)
        parser.add_argument("--sample", type=int, default=2)
        # fmt: on

    def __init__(self, model, optimizer_cfg, sample=2, temperature=0.5, lmbda=0.5):
        super(GrandModelWrapper, self).__init__(model, optimizer_cfg)
        self.sample = sample
        self.temperature = temperature
        self.lmbda = lmbda

    def train_step(self, batch):
        graph = batch
        output_list = []
        for i in range(self.sample):
            output_list.append(self.model(graph))
        loss_train = 0.0
        for output in output_list:
            loss_train += self.default_loss_fn(output[graph.train_mask], graph.y[graph.train_mask])
        loss_train = loss_train / self.sample

        if len(graph.y.shape) > 1:
            output_list = [torch.sigmoid(x) for x in output_list]
        else:
            output_list = [F.log_softmax(x, dim=-1) for x in output_list]
        loss_consis = self.consistency_loss(output_list, graph.train_mask)

        return loss_train + loss_consis

    def consistency_loss(self, logps, train_mask):
        temp = self.temperature
        ps = [torch.exp(p)[~train_mask] for p in logps]
        sum_p = 0.0
        for p in ps:
            sum_p = sum_p + p
        avg_p = sum_p / len(ps)
        sharp_p = (torch.pow(avg_p, 1.0 / temp) / torch.sum(torch.pow(avg_p, 1.0 / temp), dim=1, keepdim=True)).detach()
        loss = 0.0
        for p in ps:
            loss += torch.mean((p - sharp_p).pow(2).sum(1))
        loss = loss / len(ps)

        return self.lmbda * loss
