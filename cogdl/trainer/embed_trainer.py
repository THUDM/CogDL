import os
import numpy as np
import time
from typing import Optional
import torch


class EmbeddingTrainer(object):
    def __init__(
        self, save_emb_path: Optional[str] = None, load_emb_path: Optional[str] = None,
    ):
        self.save_emb_path = save_emb_path
        self.load_emb_path = load_emb_path
        self.default_emb_dir = "./embeddings"

    def run(self, model_w, dataset_w):
        self.prepare_data_wrapper(dataset_w)
        if self.load_emb_path is not None:
            print(f"Loading embeddings from {self.load_emb_path} ...")
            embedding = np.load(self.load_emb_path)
            return self.test(model_w, dataset_w, embedding)

        if self.save_emb_path is None:
            cur_time = time.strftime("%m-%d_%H.%M.%S", time.localtime())
            name = f"{model_w.wrapped_model.__class__.__name__}_{cur_time}.emb"
            self.save_emb_path = os.path.join(self.default_emb_dir, name)
            os.makedirs(self.default_emb_dir, exist_ok=True)
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
        train_data = dataset_w.on_train_wrapper()
        embeddings = []
        for batch in train_data:
            embeddings.append(model_w.train_step(batch))
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
        np.save(self.save_emb_path, embeddings)
