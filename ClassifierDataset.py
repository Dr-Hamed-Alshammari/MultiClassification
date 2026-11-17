import os
import json
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

dataset_folder = config['root_folder']
num_labels = config['num_labels']
max_length = config.get('max_length', 512)

# class_names order defines the label ids: index 0 -> class_names[0], etc.
class_names = [name.lower() for name in config['class_names']]
if len(class_names) != num_labels:
    raise ValueError(f"config 'class_names' has {len(class_names)} entries but 'num_labels' is {num_labels}.")
label2id = {name: idx for idx, name in enumerate(class_names)}

# Define the mapping from data_type to CSV filenames
data_type_to_filename = {
    'train': 'Training.csv',
    'val': 'Validation.csv',
    'test': 'Testing.csv'
}


def remove_arabic_diacritics(text):
    diacritic_chars = {chr(i) for i in range(0x064B, 0x065F)}
    return ''.join([char for char in text if char not in diacritic_chars])


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, data_type):
        logging.info(f'Initializing ClassifierDataset ({data_type})')

        if not os.path.isdir(dataset_folder):
            logging.error(f"Directory not found: {dataset_folder}")
            raise FileNotFoundError(f"{dataset_folder} does not exist or is not accessible.")

        self.tokenizer = tokenizer
        file_name = data_type_to_filename.get(data_type.lower())
        if not file_name:
            raise ValueError(f"Invalid data_type provided: {data_type}. Expected 'train', 'val', or 'test'.")

        file_path = os.path.join(dataset_folder, file_name)
        if not os.path.isfile(file_path):
            logging.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"{file_path} does not exist or is not accessible.")

        self.samples = []
        self.labels = []
        self._load_csv(file_path)

        if len(self.samples) == 0:
            logging.error("No valid data samples found.")
            raise RuntimeError("No valid data samples found.")

        logging.info(f"Loaded {len(self.samples)} samples from {file_name}.")
        counts = self.get_class_counts().tolist()
        logging.info("Class distribution: " + ", ".join(f"{class_names[i]}={counts[i]}" for i in range(num_labels)))
        logging.info('ClassifierDataset initialized with dediacritized text.')

    @staticmethod
    def _parse_label(raw_label, row_index):
        """Accepts either an integer id (0..num_labels-1) or a class name such as 'positive'."""
        if isinstance(raw_label, str):
            value = raw_label.strip().lower()
            if value in label2id:
                return label2id[value]
            try:
                raw_label = int(float(value))  # e.g. "2" or "2.0"
            except ValueError:
                raise ValueError(f"Row {row_index}: unknown label '{raw_label}'. Expected one of {class_names} "
                                 f"or an integer in [0, {num_labels - 1}].")
        label = int(raw_label)
        if not 0 <= label < num_labels:
            raise ValueError(f"Row {row_index}: label {label} is out of range [0, {num_labels - 1}].")
        return label

    def _load_csv(self, file_path):
        df = pd.read_csv(file_path)
        missing_columns = {'text', 'label'} - set(df.columns)
        if missing_columns:
            raise ValueError(f"{file_path} is missing column(s): {missing_columns}")

        n_before = len(df)
        df = df.dropna(subset=['text', 'label'])
        if len(df) < n_before:
            logging.warning(f"Dropped {n_before - len(df)} rows with empty text or label from {file_path}.")

        for index, row in df.iterrows():
            text = remove_arabic_diacritics(str(row['text']))
            label = self._parse_label(row['label'], index)
            self.samples.append((text, label))
            self.labels.append(label)

    def get_class_counts(self):
        """Number of samples per class, as a tensor of length num_labels."""
        return torch.bincount(torch.tensor(self.labels, dtype=torch.long), minlength=num_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )

        input_dict = {key: val.squeeze(0) for key, val in encoding.items()}
        # CrossEntropyLoss expects integer class ids (torch.long), not floats
        return input_dict, torch.tensor(label, dtype=torch.long)