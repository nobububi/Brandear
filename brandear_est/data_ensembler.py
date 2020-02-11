import numpy as np


class DataEmsembler:
    def __init__(self, ModelClass):
        self.model_class = ModelClass
        self.models = None

    def train(self, datasets):
        self.models = [self.model_class()] * len(datasets)
        [model.train(train_dataset=dataset["train_dataset"], valid_dataset=dataset["valid_dataset"])
         for model, dataset in zip(self.models, datasets)]

    def predict(self, dataset):
        return np.mean([model.predict(dataset) for model in self.models], axis=0)