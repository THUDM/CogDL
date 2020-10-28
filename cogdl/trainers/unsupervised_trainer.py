from cogdl.trainers.base_trainer import BaseTrainer


class UnsupervisedTrainer(BaseTrainer):
    def get_embedding(self):
        raise NotImplementedError("Trainers must implement the get_embedding method")
