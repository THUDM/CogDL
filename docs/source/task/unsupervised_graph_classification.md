<h3>Unsupervised graph classification</h3>

In this section, we will introduce the implementation "Unsupervised graph classification task". 

<h5>Task Design</h5>

1. Set up "UnsupervisedGraphClassification" class, which has two specific parameters.

   * `num-shuffle` : Shuffle times in classifier
   * `degree-feature`: Use one-hot node degree as node feature, for datasets such as lmdb-binary and lmdb-multi, which don't have node features.
   * `lr`: learning

   ```python
   @register_task("unsupervised_graph_classification")
   class UnsupervisedGraphClassification(BaseTask):
       r"""Unsupervised graph classification"""
       @staticmethod
       def add_args(parser):
           """Add task-specific arguments to the parser."""
           # fmt: off
           parser.add_argument("--num-shuffle", type=int, default=10)
           parser.add_argument("--degree-feature", dest="degree_feature", action="store_true")
           parser.add_argument("--lr", type=float, default=0.001)
           # fmt: on
      def __init__(self, args):
   ```

2. Build dataset and convert it to a list of `Data` defined in Cogdl.

   ```python
   dataset = build_dataset(args)
   self.label = np.array([data.y for data in dataset])
   self.data = [
   	Data(x=data.x, y=data.y, edge_index=data.edge_index, edge_attr=data.edge_attr,
   		pos=data.pos).apply(lambda x:x.to(self.device))
   		for data in dataset
   ]
   ```

3. Then we build model and can run `train` to train the model and obtain graph representation. In this part, the training process of shallow models and deep models are implemented separately.

   ```python
   self.model = build_model(args)
   self.model = self.model.to(self.device)
   
   def train(self):
   	if self.use_nn:
           # deep neura network models
   		epoch_iter = tqdm(range(self.epoch))
           for epoch in epoch_iter:
               loss_n = 0
               for batch in self.data_loader:
                   batch = batch.to(self.device)
                   predict, loss = self.model(batch.x, batch.edge_index, batch.batch)
               	self.optimizer.zero_grad()
               	loss.backward()
               	self.optimizer.step()
               	loss_n += loss.item()
   		# ... 
       else:
          # shallow models
           prediction, loss = self.model(self.data)
           label = self.label
   ```

4. When graph representation is obtained, we evaluate the embedding with `SVM` via running `num_shuffle` times under different training ratio. You can also call `save_emb` to save the embedding.

   ```python
   return self._evaluate(prediction, label)
   def _evaluate(self, embedding, labels):
       # ...
       for training_percent in training_percents:
       	for shuf in shuffles:
               # ...
               clf = SVC()
               clf.fit(X_train, y_train)
               preds = clf.predict(X_test)
               # ...
   ```

The overall implementation of UnsupervisedGraphClassification is at (https://github.com/THUDM/cogdl/blob/master/cogdl/tasks/unsupervised_graph_classification.py).

<h5>Create a model</h5>

​	To create a model for task unsupervised graph classification, the following functions have to be implemented.

1. `add_args(parser)`: add necessary hyper-parameters used in model.

   ```
   @staticmethod
   def add_args(parser):
   	 parser.add_argument("--hidden-size", type=int, default=128)
   	 parser.add_argument("--nn", type=bool, default=False)
   	 parser.add_argument("--lr", type=float, default=0.001)
   	 # ...
   ```

2. `build_model_from_args(cls, args)`: this function is called in 'task' to build model.

3. `forward`: For shallow models, this function runs as training process of model and will be called only once; For deep neural network models,  this function is actually the forward propagation process and will be called many times. 

   ```python
   # shallow model
   def forward(self, graphs):
   	# ...
       self.model = Doc2Vec(
           self.doc_collections,
   		...
       )
       vectors = np.array([self.model["g_"+str(i)] for i in range(len(graphs))])
       return vectors, None
   ```

<h5>Run</h5>

To run UnsupervisedGraphClassification, we can use the following command:

```
python scripts/train.py --task unsupervised?_graph_classification --dataset proteins --model graph2vec infograph --seed 0
```

Then We get experimental results like this:

| Variant                   | Acc           |
| ------------------------- | ------------- |
| ('proteins', 'graph2vec') | 0.7183±0.0043 |
| ('proteins', 'infograph') | 0.7396±0.0070 |
| ('proteins', 'dgk')       | 0.7354±0.0118 |
