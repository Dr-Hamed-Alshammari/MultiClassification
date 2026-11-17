import os
import json
import logging
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, auc, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize

from Classifier import CustomModel
from ClassifierDataset import ClassifierDataset, remove_arabic_diacritics

" This class is used to evaluate the model on the testing set (and to run predictions on new texts)."


class Evaluator:
    def __init__(self, model_path, model_name, num_labels, class_names, device, max_length=512):
        logging.basicConfig(filename='classifier.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Evaluator...")

        if len(class_names) != num_labels:
            raise ValueError("class_names must have exactly num_labels entries.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")

        self.num_labels = num_labels
        self.class_names = list(class_names)
        self.device = device
        self.max_length = max_length

        try:
            self.model = CustomModel(model_name, num_labels)
            # map_location lets a GPU-trained checkpoint load on a CPU-only machine
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.loss_fn = torch.nn.CrossEntropyLoss()  # Multi-class loss on raw logits
            self.logger.info(f"Model loaded for evaluation from {model_path}")
        except RuntimeError as e:
            self.logger.error(f"Checkpoint does not match a {num_labels}-class model "
                              f"(was it saved from the old binary model?). Error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model. Error: {e}")
            raise

    def evaluate(self, data_loader, predictions_csv='test_predictions.csv'):
        self.model.eval()
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []

        progress_bar = tqdm(data_loader, desc="Evaluating (testing set)", leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    inputs, labels = batch
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    labels = labels.to(self.device).long()  # Integer class ids

                    logits = self.model(input_ids=inputs['input_ids'],
                                        attention_mask=inputs['attention_mask'])  # (batch, num_labels)
                    loss = self.loss_fn(logits, labels)
                    total_loss += loss.item()

                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = probabilities.argmax(dim=-1)
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predictions.cpu().numpy())
                    y_prob.append(probabilities.cpu().numpy())
                except Exception as e:
                    # Re-raise: silently skipping a batch would make the metrics wrong
                    self.logger.error(f"Error during model evaluation: {e}", exc_info=True)
                    raise

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.concatenate(y_prob, axis=0)
        label_ids = list(range(self.num_labels))

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=label_ids, average='weighted', zero_division=0)
        try:
            auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=label_ids)
        except ValueError as e:
            self.logger.warning(f"AUC-ROC could not be computed: {e}")
            auc_score = float('nan')

        report = classification_report(y_true, y_pred, labels=label_ids, target_names=self.class_names,
                                       digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)

        self.logger.info("Testing Evaluation Metrics:")
        self.logger.info(f"  - Average Loss: {avg_loss}")
        self.logger.info(f"  - Accuracy: {accuracy}")
        self.logger.info(f"  - Macro Precision: {precision_macro}")
        self.logger.info(f"  - Macro Recall: {recall_macro}")
        self.logger.info(f"  - Macro F1: {f1_macro}")
        self.logger.info(f"  - Weighted F1: {f1_weighted}")
        self.logger.info(f"  - Macro AUC-ROC (OvR): {auc_score}")
        self.logger.info(f"  - Per-class report:\n{report}")
        self.logger.info(f"  - Confusion Matrix (rows = actual, cols = predicted, order = {self.class_names}):\n{cm}")
        print(report)

        # Save per-sample predictions (useful for error analysis)
        if predictions_csv:
            df = pd.DataFrame({
                'true_label': [self.class_names[i] for i in y_true],
                'predicted_label': [self.class_names[i] for i in y_pred],
            })
            for idx, name in enumerate(self.class_names):
                df[f'prob_{name}'] = y_prob[:, idx]
            df.to_csv(predictions_csv, index=False)
            self.logger.info(f"Per-sample predictions saved to {predictions_csv}")

        self.plot_roc_curve(y_true, y_prob)
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_normalized_confusion_matrix(y_true, y_pred)

        return {
            'loss': avg_loss, 'accuracy': accuracy, 'precision_macro': precision_macro,
            'recall_macro': recall_macro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
            'auc_roc_macro_ovr': auc_score,
        }

    def predict(self, texts, tokenizer, batch_size=64):
        """Inference on raw, unlabeled texts. Returns (predicted class names, probability array)."""
        self.model.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = [remove_arabic_diacritics(str(t)) for t in texts[start:start + batch_size]]
                encoding = tokenizer(batch_texts, truncation=True, padding=True,
                                     max_length=self.max_length, return_tensors='pt')
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                logits = self.model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        labels = [self.class_names[i] for i in probs.argmax(axis=1)]
        return labels, probs

    def plot_roc_curve(self, y_true, y_prob):
        """One-vs-rest ROC curve for each class."""
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_labels)))
        plt.figure(figsize=(8, 6))
        for class_idx, class_name in enumerate(self.class_names):
            positives = y_true_bin[:, class_idx].sum()
            if positives == 0 or positives == len(y_true_bin):
                continue  # ROC is undefined if the class is absent (or the only class present)
            fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Testing ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curve_testing.png')
        plt.show()

    def _plot_cm(self, values, title, cmap, fmt, file_name):
        plt.figure(figsize=(7, 6))
        plt.imshow(values, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.num_labels)
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        thresh = values.max() / 2.
        for i, j in product(range(values.shape[0]), range(values.shape[1])):
            plt.text(j, i, format(values[i, j], fmt), horizontalalignment="center",
                     color="white" if values[i, j] > thresh else "black")
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.tight_layout()
        plt.savefig(file_name)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_labels)))
        self._plot_cm(cm, 'Testing Confusion Matrix', plt.cm.Greens, 'd', 'confusion_matrix_eval.png')

    def plot_normalized_confusion_matrix(self, y_true, y_pred):
        """Row-normalized: the diagonal shows each class's recall."""
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_labels)))
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float),
                                  where=row_sums != 0)
        self._plot_cm(cm_normalized, 'Normalized Testing Confusion Matrix', plt.cm.PuBuGn, '.2f',
                      'confusion_matrix_eval_normalized.png')


# Function to run the evaluation independently
def run_evaluation():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = ClassifierDataset(tokenizer, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config['testing_batch_size'], shuffle=False)

    best_model_path = os.path.join(config['checkpoint_path'], "best_model.pt")
    evaluator = Evaluator(
        best_model_path,
        model_name,
        num_labels=config['num_labels'],
        class_names=config['class_names'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        max_length=config.get('max_length', 512),
    )

    evaluator.evaluate(test_loader)

    # Example: inference on new, unlabeled texts
    # labels, probs = evaluator.predict(["الخدمة ممتازة جدا", "المنتج سيء"], tokenizer)
    # print(labels, probs)


if __name__ == '__main__':
    run_evaluation()