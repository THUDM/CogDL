import torch
from cogdl.utils import edge_softmax, batch_sum_pooling


class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_feats, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_feats
        self.out_channels = 2 * in_feats
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels, num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1

        h = (
            x.new_zeros((self.num_layers, batch_size, self.in_channels)),
            x.new_zeros((self.num_layers, batch_size, self.in_channels)),
        )
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = edge_softmax(e, batch)
            r = batch_sum_pooling(a * x, batch)
            q_star = torch.cat([q, r], dim=-1)

        return q_star
