import os
import logging

import numpy as np
import torch
from torch.autograd import Variable


# ===========
# function
# ===========

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)


def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)


# =========
# class
# =========

class TopAverage(object):
    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_top_average(self):
        if len(self.scores) > 0:
            return np.mean(self.scores)
        else:
            return 0

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        # print("Top %d average: %f" % (self.top_k, avg))
        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)


class FixedList(list):
    def __init__(self, size=10):
        super(FixedList, self).__init__()
        self.size = size

    def append(self, obj):
        if len(self) >= self.size:
            self.pop(0)
        super().append(obj)


class EarlyStop(object):
    def __init__(self, size=10):
        self.size = size
        self.train_loss_list = FixedList(size)
        self.train_score_list = FixedList(size)
        self.val_loss_list = FixedList(size)
        self.val_score_list = FixedList(size)

    def should_stop(self, train_loss, train_score, val_loss, val_score):
        flag = False
        if len(self.train_loss_list) < self.size:
            pass
        else:
            if val_loss > 0:
                if val_loss >= np.mean(self.val_loss_list):  # and val_score <= np.mean(self.val_score_list)
                    flag = True
            elif train_loss > np.mean(self.train_loss_list):
                flag = True
        self.train_loss_list.append(train_loss)
        self.train_score_list.append(train_score)
        self.val_loss_list.append(val_loss)
        self.val_score_list.append(val_score)

        return flag

    def should_save(self, train_loss, train_score, val_loss, val_score):
        if len(self.val_loss_list) < 1:
            return False
        if train_loss < min(self.train_loss) and val_score > max(self.val_score_list):
            # if val_loss < min(self.val_loss_list) and val_score > max(self.val_score_list):
            return True
        else:
            return False