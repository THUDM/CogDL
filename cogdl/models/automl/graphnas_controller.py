import torch
import torch.nn.functional as F

import cogdl.models.automl.utils as utils


# not contains skip-connection
class SimpleNASController(torch.nn.Module):
    def _construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            for action, action_name in zip(single_action, self.action_list):
                predicted_actions = self.search_space[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def __init__(self, args, search_space, action_list, controller_hid=100, cuda=True, mode="train",
                 softmax_temperature=5.0, tanh_c=2.5):
        print(action_list, search_space.keys())
        if not self.check_action_list(action_list, search_space):
            raise RuntimeError("There are actions not contained in search_space")
        super(SimpleNASController, self).__init__()
        self.mode = mode
        # search space or operators set containing operators used to build GNN
        self.search_space = search_space
        # operator categories for each controller RNN output
        self.action_list = action_list
        self.controller_hid = controller_hid
        self.is_cuda = cuda

        # set hyperparameters
        if args and args.softmax_temperature:
            self.softmax_temperature = args.softmax_temperature
        else:
            self.softmax_temperature = softmax_temperature
        if args and args.tanh_c:
            self.tanh_c = args.tanh_c
        else:
            self.tanh_c = tanh_c

        # build encoder
        self.num_tokens = []
        for key in self.search_space:
            self.num_tokens.append(len(self.search_space[key]))

        num_total_tokens = sum(self.num_tokens)  # count action type
        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)

        # the core of controller
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        # build decoder
        self._decoders = torch.nn.ModuleDict()
        for key in self.search_space:
            size = len(self.search_space[key])
            decoder = torch.nn.Linear(controller_hid, size)
            self._decoders[key] = decoder

        self.reset_parameters()

    # use to scale up the predicted network
    def update_action_list(self, action_list):
        if not self.check_action_list(action_list, self.search_space):
            raise RuntimeError("There are actions not contained in search_space")

        self.action_list = action_list

    @staticmethod
    def check_action_list(action_list, search_space):
        if isinstance(search_space, dict):
            keys = search_space.keys()
        else:
            return False
        for each in action_list:
            if each in keys:
                pass
            else:
                return False
        return True

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self._decoders:
            self._decoders[decoder].bias.data.fill_(0)

    def forward(self,
                inputs,
                hidden,
                action_name,
                is_embed):

        embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self._decoders[action_name](hx)

        logits /= self.softmax_temperature

        # exploration
        if self.mode == 'train':
            logits = (self.tanh_c * torch.tanh(logits))

        return logits, (hx, cx)

    def action_index(self, action_name):
        key_names = self.search_space.keys()
        for i, key in enumerate(key_names):
            if action_name == key:
                return i

    def sample(self, batch_size=1, with_details=False):

        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        inputs = torch.zeros([batch_size, self.controller_hid])
        hidden = (torch.zeros([batch_size, self.controller_hid]), torch.zeros([batch_size, self.controller_hid]))
        if self.is_cuda:
            inputs = inputs.cuda()
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        entropies = []
        log_probs = []
        actions = []
        for block_idx, action_name in enumerate(self.action_list):
            decoder_index = self.action_index(action_name)

            logits, hidden = self.forward(inputs,
                                          hidden,
                                          action_name,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:decoder_index]),
                self.is_cuda,
                requires_grad=False)

            inputs = self.encoder(inputs)

            actions.append(action[:, 0])

        actions = torch.stack(actions).transpose(0, 1)
        dags = self._construct_action(actions)

        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.is_cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.is_cuda, requires_grad=False))


