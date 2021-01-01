Trainer
=========

In this section, we will introduce how to implement a specific `Trainer` for a model.

In previous section, we introduce the implementation of different `tasks`. But the training paradigm varies and is
incompatible with the defined training process in some cases. Therefore, `CogDL` provides `Trainer` to customize the
training and inference mode. Take `NeighborSamplingTrainer` as the example, this section will show how to define a trainer.

**Design**

1. A self-defined trainer should inherits `BaseTrainer` and must implement function `fit` to define the training and
evaluating process. Necessary parameters for training need to be added to the `add_args` in models and can be obtained here in `__init___`.

.. code-block:: python

    class NeighborSamplingTrainer(BaseTrainer):
        def __init__(self, args):
            # ... get necessary parameters from args

        def fit(self, model, dataset):
            # ... implement the training and evaluation

        @classmethod
        def build_trainer_from_args(cls, args):
            return cls(args)


2. All training and evaluating process, including data preprocessing and defining optimizer, should be implemented in `fit`.
In other words, given the model and dataset, the rest is up to you. `fit` accepts two parameters: model and dataset, which
usually are in cpu. You need to move them to cuda if you want to train on GPU.

.. code-block:: python

    def fit(self, model, dataset):
        self.data = dataset[0]

        # preprocess data
        self.train_loader = NeighborSampler(
            data=self.data,
            mask=self.data.train_mask,
            sizes=self.sample_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        self.test_loader = NeighborSampler(
            data=self.data, mask=None, sizes=[-1], batch_size=self.batch_size, shuffle=False
        )
        # move model to GPU
        self.model = model.to(self.device)

        # define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # training
        best_model = self.train()
        self.model = best_model
        # evaluation
        acc, loss = self._test_step()
        return dict(Acc=acc["test"], ValAcc=acc["val"])

3. To make the training of a model use the trainer, we should assign the trainer to the model. In Cogdl, a model must implement
`get_trainer` as static method if it has a customized training process.
GraphSAGE depends on `NeighborSamplingTrainer`, so the following codes should exsits in the implementation.

.. code-block:: python

    @staticmethod
    def get_trainer(taskType, args):
        return NeighborSamplingTrainer

The details of training and evaluating are similar to the implementation in `Tasks`. The overall implementation of trainers is at
https://github.com/THUDM/cogdl/tree/master/cogdl/trainers
