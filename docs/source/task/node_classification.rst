Node Classification
===================

In this tutorial, we will introduce a important task, node classification. In this task, we train a GNN model with partial node labels and use accuracy to measure the performance. 

**Semi-supervied Node Classification Methods**

+------------+----------+-----------+-----------------+
| Method     | Sampling | Inductive | Reproducibility |
+============+==========+===========+=================+
| GCN        |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| GAT        |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| Chebyshev  |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| GraphSAGE  |  `√`     |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| GRAND      |          |           |    `√`          |
+------------+----------+-----------+-----------------+
| GCNII      |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| DeeperGCN  |  `√`     |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| Dr-GAT     |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| U-net      |          |           |    `√`          |
+------------+----------+-----------+-----------------+
| APPNP      |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| GraphMix   |          |           |    `√`          |
+------------+----------+-----------+-----------------+
| DisenGCN   |          |           |                 |
+------------+----------+-----------+-----------------+
| SGC        |          |   `√`     |   `√`           |
+------------+----------+-----------+-----------------+
| JKNet      |          |   `√`     |   `√`           |
+------------+----------+-----------+-----------------+
| MixHop     |          |           |                 |
+------------+----------+-----------+-----------------+
| DropEdge   |   `√`    |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+
| SRGCN      |          |   `√`     |    `√`          |
+------------+----------+-----------+-----------------+

.. tip::

    Reproducibility means whether the model is reproduced in our experimental setting currently.

First we define the `NodeClassification` class.

.. code-block:: python

    @register_task("node_classification")
    class NodeClassification(BaseTask):
        """Node classification task."""

        @staticmethod
        def add_args(parser):
            """Add task-specific arguments to the parser."""

        def __init__(self, args):
            super(NodeClassification, self).__init__(args)

Then we can build dataset and model according to args. Generally the model and dataset should be placed in the same
device using `.to(device)` instead of `.cuda()`. And then we set the optimizer.

.. code-block:: python

    self.device = torch.device('cpu' if args.cpu else 'cuda')
    # build dataset with `build_dataset`
    dataset = build_dataset(args)
    self.data = dataset.data
    self.data.apply(lambda x: x.to(self.device))
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes

    # build model with `build_model`
    model = build_model(args)
    self.model = model.to(self.device)
    self.patience = args.patience
    self.max_epoch = args.max_epoch

    # set optimizer
    self.optimizer = torch.optim.Adam(
        self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

For the training process, `train` must be implemented as it will be called as the entrance of training.
We provide a training loop for node classification task. For each epoch, we first call `_train_step` to optimize our
model and then call `_test_step` for validation and test to compute the accuracy and loss.

.. code-block:: python

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )

    def _train_step(self):
        """train step per epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        # In node classification task, `node_classification_loss` must be defined in model if you want to use this task directly.
        self.model.node_classification_loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        """test step"""
        self.model.eval()
        # `Predict` should be defined in model for inference.
        logits = self.model.predict(self.data)
        logits = F.log_softmax(logits, dim=-1)
        mask = self.data.test_mask
        loss = F.nll_loss(logits[mask], self.data.y[mask]).item()

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss

In supervied node classification tasks, we use early stopping to reduce over-fitting and save training time.

.. code-block:: python

    if val_loss <= min_loss or val_acc >= max_score:
        if val_loss <= best_loss:  # and val_acc >= best_score:
            best_loss = val_loss
            best_score = val_acc
            best_model = copy.deepcopy(self.model)
        min_loss = np.min((min_loss, val_loss))
        max_score = np.max((max_score, val_acc))
        patience = 0
    else:
        patience += 1
        if patience == self.patience:
            self.model = best_model
            epoch_iter.close()
            break

Finally, we compute the accuracy scores of test set for the trained model.

.. code-block:: python

    test_acc, _ = self._test_step(split="test")
    print(f"Test accuracy = {test_acc}")
    return dict(Acc=test_acc)

The overall implementation of `NodeClassification` is at (https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/node_classification.py).

To run NodeClassification, we can use the following command:

.. code-block:: python

    python scripts/train.py --task node_classification --dataset cora citeseer --model gcn gat --seed 0 1 --max-epoch 500


Then We get experimental results like this:

=========================  ============== 
Variant                    Acc   
=========================  ==============
('cora', 'gcn')            0.8220±0.0010
('cora', 'gat')            0.8275±0.0015
('citeseer', 'gcn')        0.7060±0.0050
('citeseer', 'gat')        0.7060±0.0020
=========================  ==============
