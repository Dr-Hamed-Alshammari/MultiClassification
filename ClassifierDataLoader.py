import logging

import torch
from torch.utils.data import Dataset, DataLoader


class ClassifierDataLoader:

    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, seed,
                 test_batch_size=None, num_workers=0, pin_memory=False):
        # Ensure the datasets provided are of type `Dataset`
        if not all(isinstance(dataset, Dataset) for dataset in (train_dataset, val_dataset, test_dataset)):
            raise TypeError("All provided datasets must be of type `Dataset`.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        # Falls back to batch_size if no separate testing batch size is given
        self.test_batch_size = test_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    def get_data_loaders(self):
        logging.info("Creating training data loader.")
        g = torch.Generator()
        g.manual_seed(self.seed)  # Use the seed passed in the constructor
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=g
        )

        logging.info("Creating validation data loader.")
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        logging.info("Creating test data loader.")
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return train_loader, val_loader, test_loader