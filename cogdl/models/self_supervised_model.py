from cogdl.models import BaseModel
from abc import abstractmethod


class SelfSupervisedModel(BaseModel):
    @abstractmethod
    def self_supervised_loss(self, data):
        raise NotImplementedError

    @staticmethod
    def get_trainer(args):
        return None


class SelfSupervisedGenerativeModel(SelfSupervisedModel):
    @abstractmethod
    def generate_virtual_labels(self, data):
        raise NotImplementedError


class SelfSupervisedContrastiveModel(SelfSupervisedModel):
    @abstractmethod
    def augment(self, data):
        raise NotImplementedError
