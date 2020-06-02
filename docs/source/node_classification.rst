Node Classification
===================

In this tutorial, we will introduce a important task, node classification. In this task, we train a GNN model with partial node labels and use accuracy to measure the performance. 

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

Then we can build dataset according to args.

.. code-block:: python

    self.device = torch.device('cpu' if args.cpu else 'cuda')
    dataset = build_dataset(args)
    self.data = dataset.data
    self.data.apply(lambda x: x.to(self.device))
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes

After that, we can build model and use `Adam` to optimize the model.

.. code-block:: python

    model = build_model(args)
    self.model = model.to(self.device)
    self.patience = args.patience
    self.max_epoch = args.max_epoch
    self.optimizer = torch.optim.Adam(
        self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

We provide a training loop for node classification task. For each epoch, we first call `_train_step` to optimize our model and then call `_test_step` to compute the accuracy and loss.

.. code-block:: python

    def train(self):
        epoch_iter = tqdm(range(self.max_epoch))
        patience = 0
        best_score = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in epoch_iter:
            self._train_step()
            train_acc, _ = self._test_step(split="train")
            val_acc, val_loss = self._test_step(split="val")
            epoch_iter.set_description(
                f"Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )
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
    
    def _train_step(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.loss(self.data).backward()
        self.optimizer.step()

    def _test_step(self, split="val"):
        self.model.eval()
        logits = self.model.predict(self.data)
        _, mask = list(self.data(f"{split}_mask"))[0]
        loss = F.nll_loss(logits[mask], self.data.y[mask])

        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc, loss


Finally, we compute the accuracy scores of test set for the trained model.

.. code-block:: python

    test_acc, _ = self._test_step(split="test")
    print(f"Test accuracy = {test_acc}")
    return dict(Acc=test_acc)

The overall implementation of `NodeClassification` is at (https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/node_classification.py).

To run NodeClassification, we can use the following command:

.. code-block:: python

    python scripts/train.py --task node_classification --dataset cora citeseer --model pyg_gcn pyg_gat --seed 0 1 --max-epoch 500


Then We get experimental results like this:

=========================  ============== 
Variant                    Acc   
=========================  ============== 
('cora', 'pyg_gcn')        0.7785±0.0165  
('cora', 'pyg_gat')        0.7925±0.0045  
('citeseer', 'pyg_gcn')    0.6535±0.0195 
('citeseer', 'pyg_gat')    0.6675±0.0025 
=========================  ============== 
