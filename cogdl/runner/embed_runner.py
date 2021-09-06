import os
import numpy as np
import time
from typing import Optional
import torch


class EmbeddingTrainer(object):
    def __init__(
        self,
        save_embedding_path: Optional[str] = None,
        load_embedding_path: Optional[str] = None,
    ):
        self.save_embedding_path = save_embedding_path
        self.default_embedding_dir = "./embeddings"
        self.load_embedding_path = load_embedding_path

    def run(self, model_w, dataset_w):
        self.prepare_data_wrapper(dataset_w)
        if self.load_embedding_path is not None:
            embedding = np.load(self.load_embedding_path)
            return self.test(model_w, dataset_w, embedding)

        if self.save_embedding_path is None:
            cur_time = time.strftime("%m-%d_%H.%M.%S", time.localtime())
            name = f"{model_w.wrapped_model.__class__.__name__}_{cur_time}.emb"
            self.save_embedding_path = os.path.join(self.default_embedding_dir, name)
            os.makedirs(self.default_embedding_dir, exist_ok=True)
        embeddings = self.train(model_w, dataset_w)
        self.save_embedding(embeddings)
        return self.test(model_w, dataset_w, embeddings)

    def prepare_data_wrapper(self, dataset_w):
        dataset_w.pre_transform()
        dataset_w.prepare_training_data()
        dataset_w.prepare_val_data()
        dataset_w.prepare_test_data()

    def train(self, model_w, dataset_w):
        dataset_w.pre_transform()
        train_data = dataset_w.on_training_wrapper()
        embeddings = []
        for batch in train_data:
            embeddings.append(model_w.train_step(batch))
        # embeddings = model_w.train_step(train_data)
        assert len(embeddings) == 1
        embeddings = embeddings[0]
        return embeddings

    def test(self, model_w, dataset_w, embeddings):
        labels = next(dataset_w.on_test_wrapper())
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        result = model_w.test_step((embeddings, labels))
        return result

    def save_embedding(self, embeddings):
        np.save(self.save_embedding_path, embeddings)
