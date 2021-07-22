import copy
from torch.utils.checkpoint import get_device_states, set_device_states


# # Code borrowed from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
# # following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
# class Deterministic(nn.Module):
#     def __init__(self, net):
#         super().__init__()
#         self.net = net
#         self.cpu_state = None
#         self.cuda_in_fwd = None
#         self.gpu_devices = None
#         self.gpu_states = None
#
#     def record_rng(self, *args):
#         self.cpu_state = torch.get_rng_state()
#         if torch.cuda._initialized:
#             self.cuda_in_fwd = True
#             self.gpu_devices, self.gpu_states = get_device_states(*args)
#
#     def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
#         if record_rng:
#             self.record_rng(*args)
#
#         if not set_rng:
#             return self.net(*args, **kwargs)
#
#         rng_devices = []
#         if self.cuda_in_fwd:
#             rng_devices = self.gpu_devices
#
#         with torch.random.fork_rng(devices=rng_devices, enabled=True):
#             torch.set_rng_state(self.cpu_state)
#             if self.cuda_in_fwd:
#                 set_device_states(self.gpu_devices, self.gpu_states)
#             return self.net(*args, **kwargs)
#
#
# # Code borrowed from https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
# # once multi-GPU is confirmed working, refactor and send PR back to source
# class ReversibleBlock(nn.Module):
#     def __init__(self, f, g, depth=None, send_signal=False):
#         super().__init__()
#         self.f = Deterministic(f)
#         self.g = Deterministic(g)
#
#         self.depth = depth
#         self.send_signal = send_signal
#
#     def forward(self, x, f_args: Dict = {}, g_args: Dict = {}, dim=1):
#         x1, x2 = torch.chunk(x, 2, dim=dim)
#         y1, y2 = None, None
#
#         if self.send_signal:
#             f_args["_reverse"] = g_args["_reverse"] = False
#             f_args["_depth"] = g_args["_depth"] = self.depth
#
#         with torch.no_grad():
#             y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
#             y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
#
#         return torch.cat([y1, y2], dim=2)
#
#     def backward_pass(self, y, dy, f_args={}, g_args={}, dim=1):
#         y1, y2 = torch.chunk(y, 2, dim=dim)
#         del y
#
#         dy1, dy2 = torch.chunk(dy, 2, dim=dim)
#         del dy
#
#         if self.send_signal:
#             f_args["_reverse"] = g_args["_reverse"] = True
#             f_args["_depth"] = g_args["_depth"] = self.depth
#
#         with torch.enable_grad():
#             y1.requires_grad = True
#             gy1 = self.g(y1, set_rng=True, **g_args)
#             torch.autograd.backward(gy1, dy2)
#
#         with torch.no_grad():
#             x2 = y2 - gy1
#             del y2, gy1
#
#             dx1 = dy1 + y1.grad
#             del dy1
#             y1.grad = None
#
#         with torch.enable_grad():
#             x2.requires_grad = True
#             fx2 = self.f(x2, set_rng=True, **f_args)
#             torch.autograd.backward(fx2, dx1, retain_graph=True)
#
#         with torch.no_grad():
#             x1 = y1 - fx2
#             del y1, fx2
#
#             dx2 = dy2 + x2.grad
#             del dy2
#             x2.grad = None
#
#             x = torch.cat([x1, x2.detach()], dim=2)
#             dx = torch.cat([dx1, dx2], dim=2)
#
#         return x, dx
#
#
# class _ReversibleFunction(Function):
#     @staticmethod
#     def forward(ctx, x, blocks, kwargs):
#         ctx.kwargs = kwargs
#         for block in blocks:
#             x = block(x, **kwargs)
#         ctx.y = x.detach()
#         ctx.blocks = blocks
#         return x
#
#     @staticmethod
#     def backward(ctx, dy):
#         y = ctx.y
#         kwargs = ctx.kwargs
#         for block in ctx.blocks[::-1]:
#             y, dy = block.backward_pass(y, dy, **kwargs)
#         return dy, None, None


"""
    Code Borrowed from https://github.com/lightaime/deep_gcns_torch/blob/master/eff_gcn_modules/rev/gcn_revop.py and
    https://github.com/silvandeleemput/memcnn/blob/master/memcnn/models/revop.py
"""


import numpy as np
import torch
import torch.nn as nn


use_context_mans = True


class InvertibleCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, fn_inverse, keep_input, num_bwd_passes, preserve_rng_state, num_inputs, *inputs_and_weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.keep_input = keep_input
        ctx.weights = inputs_and_weights[num_inputs:]
        ctx.num_bwd_passes = num_bwd_passes
        ctx.preserve_rng_state = preserve_rng_state
        ctx.num_inputs = num_inputs
        inputs = inputs_and_weights[:num_inputs]

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*inputs)

        ctx.input_requires_grad = [element.requires_grad for element in inputs]

        with torch.no_grad():
            # Makes a detached copy which shares the storage
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                else:
                    x.append(element)
            outputs = ctx.fn(*x)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        # Detaches y in-place (inbetween computations can now be discarded)
        detached_outputs = tuple([element.detach_() for element in outputs])

        # clear memory from inputs
        # only clear memory of node features
        if not ctx.keep_input:
            # PyTorch 1.0+ way to clear storage for node features
            inputs[0].storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs] * num_bwd_passes
        ctx.outputs = [detached_outputs] * num_bwd_passes

        return detached_outputs

    @staticmethod
    def backward(ctx, *grad_outputs):  # pragma: no cover
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "InvertibleCheckpointFunction is not compatible with .grad(), please use .backward() if possible"
            )
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError(
                "Trying to perform backward on the InvertibleCheckpointFunction for more than "
                "{} times! Try raising `num_bwd_passes` by one.".format(ctx.num_bwd_passes)
            )
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # recompute input if necessary
        if not ctx.keep_input:
            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  Restore the surrounding state
            # when we're done.
            rng_devices = []
            if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
                rng_devices = ctx.fwd_gpu_devices
            with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
                if ctx.preserve_rng_state:
                    torch.set_rng_state(ctx.fwd_cpu_state)
                    if ctx.had_cuda_in_fwd:
                        set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
                # recompute input
                with torch.no_grad():
                    # edge_index and edge_emb
                    inputs_inverted = ctx.fn_inverse(*(outputs + inputs[1:]))
                    # clear memory from outputs
                    # PyTorch 1.0+ way to clear storage
                    for element in outputs:
                        element.storage().resize_(0)

                    if not isinstance(inputs_inverted, tuple):
                        inputs_inverted = (inputs_inverted,)
                    for element_original, element_inverted in zip(inputs, inputs_inverted):
                        element_original.storage().resize_(int(np.prod(element_original.size())))
                        element_original.set_(element_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    detached_inputs.append(element.detach())
                else:
                    detached_inputs.append(element)
            detached_inputs = tuple(detached_inputs)
            for det_input, requires_grad in zip(detached_inputs, ctx.input_requires_grad):
                det_input.requires_grad = requires_grad
            temp_output = ctx.fn(*detached_inputs)
        if not isinstance(temp_output, tuple):
            temp_output = (temp_output,)

        filtered_detached_inputs = tuple(filter(lambda x: x.requires_grad, detached_inputs))
        gradients = torch.autograd.grad(
            outputs=temp_output, inputs=filtered_detached_inputs + ctx.weights, grad_outputs=grad_outputs
        )

        # Setting the gradients manually on the inputs and outputs (mimic backwards)

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights) :]

        return (None, None, None, None, None, None) + gradients


class InvertibleModuleWrapper(nn.Module):
    def __init__(
        self, fn, keep_input=False, keep_input_inverse=False, num_bwd_passes=1, disable=False, preserve_rng_state=False
    ):
        """
        The InvertibleModuleWrapper which enables memory savings during training by exploiting
        the invertible properties of the wrapped module.
        Parameters
        ----------
            fn : :obj:`torch.nn.Module`
                A torch.nn.Module which has a forward and an inverse function implemented with
                :math:`x == m.inverse(m.forward(x))`
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.
            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.
            num_bwd_passes :obj:`int`, optional
                Number of backward passes to retain a link with the output. After the last backward pass the output
                is discarded and memory is freed.
                Warning: if this value is raised higher than the number of required passes memory will not be freed
                correctly anymore and the training process can quickly run out of memory.
                Hence, The typical use case is to keep this at 1, until it raises an error for raising this value.
            disable : :obj:`bool`, optional
                This will disable using the InvertibleCheckpointFunction altogether.
                Essentially this renders the function as `y = fn(x)` without any of the memory savings.
                Setting this to true will also ignore the keep_input and keep_input_inverse properties.
            preserve_rng_state : :obj:`bool`, optional
                Setting this will ensure that the same RNG state is used during reconstruction of the inputs.
                I.e. if keep_input = False on forward or keep_input_inverse = False on inverse. By default
                this is False since most invertible modules should have a valid inverse and hence are
                deterministic.
        Attributes
        ----------
            keep_input : :obj:`bool`, optional
                Set to retain the input information on forward, by default it can be discarded since it will be
                reconstructed upon the backward pass.
            keep_input_inverse : :obj:`bool`, optional
                Set to retain the input information on inverse, by default it can be discarded since it will be
                reconstructed upon the backward pass.
        """
        super(InvertibleModuleWrapper, self).__init__()
        self.disable = disable
        self.keep_input = keep_input
        self.keep_input_inverse = keep_input_inverse
        self.num_bwd_passes = num_bwd_passes
        self.preserve_rng_state = preserve_rng_state
        self._fn = fn

    def forward(self, *xin):
        """Forward operation :math:`R(x) = y`
        Parameters
        ----------
            *xin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).
        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *y.
        """
        if not self.disable:
            y = InvertibleCheckpointFunction.apply(
                self._fn.forward,
                self._fn.inverse,
                self.keep_input,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(xin),
                *(xin + tuple([p for p in self._fn.parameters() if p.requires_grad])),
            )
        else:
            y = self._fn(*xin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y

    def inverse(self, *yin):
        """Inverse operation :math:`R^{-1}(y) = x`
        Parameters
        ----------
            *yin : :obj:`torch.Tensor` tuple
                Input torch tensor(s).
        Returns
        -------
            :obj:`torch.Tensor` tuple
                Output torch tensor(s) *x.
        """
        if not self.disable:
            x = InvertibleCheckpointFunction.apply(
                self._fn.inverse,
                self._fn.forward,
                self.keep_input_inverse,
                self.num_bwd_passes,
                self.preserve_rng_state,
                len(yin),
                *(yin + tuple([p for p in self._fn.parameters() if p.requires_grad])),
            )
        else:
            x = self._fn.inverse(*yin)

        # If the layer only has one input, we unpack the tuple again
        if isinstance(x, tuple) and len(x) == 1:
            return x[0]
        return x


# ---------------------
# Additive Coupling
# ---------------------


class AdditiveCoupling(nn.Module):
    def __init__(self, fm, gm=None, split_dim=-1):
        """
        This computes the output :math:`y` on forward given input :math:`x` and arbitrary modules :math:`Fm` and :math:`Gm` according to:
        :math:`(x1, x2) = x`
        :math:`y1 = x1 + Fm(x2)`
        :math:`y2 = x2 + Gm(y1)`
        :math:`y = (y1, y2)`
        Parameters
        ----------
            Fm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
            Gm : :obj:`torch.nn.Module`
                A torch.nn.Module encapsulating an arbitrary function
                (If not specified a deepcopy of Fm is used as a Module)
            implementation_fwd : :obj:`int`
                Switch between different Additive Operation implementations for forward pass. Default = -1
            implementation_bwd : :obj:`int`
                Switch between different Additive Operation implementations for inverse pass. Default = -1
            split_dim : :obj:`int`
                Dimension to split the input tensors on. Default = 1, generally corresponding to channels.
        """
        super(AdditiveCoupling, self).__init__()
        # mirror the passed module, without parameter sharing...
        if fm is None:
            gm = copy.deepcopy(fm)
        self.gm = gm
        self.fm = fm
        self.split_dim = split_dim

    def forward(self, x, graph):
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        x1, x2 = x1.contiguous(), x2.contiguous()
        fmd = self.fm.forward(graph, x2)
        y1 = x1 + fmd
        gmd = self.gm.forward(graph, y1)
        y2 = x2 + gmd
        out = torch.cat([y1, y2], dim=self.split_dim)
        return out

    def inverse(self, y, graph):
        y1, y2 = torch.chunk(y, 2, dim=self.split_dim)
        y1, y2 = y1.contiguous(), y2.contiguous()
        gmd = self.gm.forward(graph, y1)
        x2 = y2 - gmd
        fmd = self.fm.forward(graph, x2)
        x1 = y1 - fmd
        x = torch.cat([x1, x2], dim=self.split_dim)
        return x


class GroupAdditiveCoupling(torch.nn.Module):
    def __init__(self, func_modules, split_dim=-1, group=2):
        super(GroupAdditiveCoupling, self).__init__()

        self.func_modules = func_modules
        self.split_dim = split_dim
        self.group = group

    def forward(self, x, graph, *args):
        xs = torch.chunk(x, self.group, dim=self.split_dim)
        if len(args) > 0:
            chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
            args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            if len(args) > 0:
                fmd = self.func_modules[i].forward(graph, y_in, *args_chunks[i])
            else:
                fmd = self.func_modules[i].forward(graph, y_in)

            y = xs[i] + fmd
            y_in = y
            ys.append(y)

        out = torch.cat(ys, dim=self.split_dim)

        return out

    def inverse(self, y, graph, *args):
        ys = torch.chunk(y, self.group, dim=self.split_dim)
        if len(args) > 0:
            chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
            args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.group - 1, -1, -1):
            if i != 0:
                y_in = ys[i - 1]
            else:
                y_in = sum(xs)

            if len(args) > 0:
                fmd = self.func_modules[i].forward(graph, y_in, *args_chunks[i])
            else:
                fmd = self.func_modules[i].forward(graph, y_in)
            x = ys[i] - fmd
            xs.append(x)

        x = torch.cat(xs[::-1], dim=self.split_dim)
        return x


# RevGNN BaseBlock
class RevGNNLayer(nn.Module):
    def __init__(self, conv, group):
        super(RevGNNLayer, self).__init__()
        self.groups = nn.ModuleList()
        for i in range(group):
            if i == 0:
                self.groups.append(conv)
            else:
                self.groups.append(copy.deepcopy(conv))
        inv_module = GroupAdditiveCoupling(self.groups, group=group)
        self.nn = InvertibleModuleWrapper(fn=inv_module, keep_input=False)

    def forward(self, *args, **kwargs):
        items = list(args)
        items[1], items[0] = items[0], items[1]
        return self.nn(*items, **kwargs)
