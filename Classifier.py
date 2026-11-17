import os
import itertools
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, auc, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize

from utils import finish_figure


class CustomClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.1):
        super(CustomClassifierHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)  # One logit per class

    def forward(self, transformer_output):
        x = self.dropout(transformer_output)
        x = torch.tanh(self.dense(x))  # Non-linearity (as in the standard RoBERTa head)
        x = self.dropout(x)
        logits = self.out_proj(x)  # Shape: (batch_size, num_labels)
        return logits


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(CustomModel, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.custom_head = CustomClassifierHead(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = transformer_outputs.last_hidden_state[:, 0, :]  # <s> (CLS) token
        final_logits = self.custom_head(hidden_state)
        return final_logits


class Classifier(nn.Module):
    def __init__(self, model_name, num_labels, class_names, learning_rate, epochs, checkpoint_path, patience,
                 initial_learning_rate, warmup_epochs, class_weights=None, show_plots=False, results_path='results'):
        super(Classifier, self).__init__()
        if len(class_names) != num_labels:
            raise ValueError("class_names must have exactly num_labels entries.")

        self.num_labels = num_labels
        self.class_names = list(class_names)
        self.model = CustomModel(model_name, num_labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger(__name__)
        self.checkpoint_path = checkpoint_path
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.last_metrics = {}

        # Multi-class loss: takes raw logits (batch, num_labels) and integer labels (batch,)
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.patience = patience
        self.early_stopping_triggered = False
        self.initial_learning_rate = initial_learning_rate
        self.warmup_epochs = warmup_epochs
        # Plots are always saved to results_path; show_plots only controls whether windows pop up
        # (and even then only for the final test-set plots, so training never pauses).
        self.show_plots = show_plots
        self.results_path = results_path

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    # ----------------------------------------------------------------Training and Evaluation Functions-------------------------------------------------------------
    def load_best_model(self):
        model_path = os.path.join(self.checkpoint_path, "best_model.pt")
        if not os.path.exists(model_path):
            self.logger.info("No best model checkpoint found.")
            return False
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Happens e.g. when the checkpoint was saved from the old binary (1-output) model
            self.logger.warning(f"Checkpoint {model_path} does not match the current model and was ignored: {e}")
            return False
        self.logger.info(f"Loaded best model from {model_path}")
        return True

    def _create_scheduler(self, steps_per_epoch):
        """Linear warmup from initial_learning_rate to learning_rate, then cosine decay to 0.
        The scheduler is stepped once per batch, so everything is expressed in optimizer steps.
        LambdaLR multiplies the base LR by the returned factor, so the factor must be a ratio."""
        total_steps = max(1, self.epochs * steps_per_epoch)
        warmup_steps = min(self.warmup_epochs * steps_per_epoch, total_steps)
        main_lr = self.optimizer.param_groups[0]['lr']
        start_factor = self.initial_learning_rate / main_lr

        def lr_lambda(step):
            if step < warmup_steps:
                return start_factor + (1.0 - start_factor) * step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0)))

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train(self, train_loader, val_loader, test_loader, start_epoch=0):
        best_val_loss = float('inf')
        epochs_without_improvement = 0  # Counter for early stopping
        final_train_accuracy = 0

        if not isinstance(train_loader, DataLoader) or not isinstance(val_loader, DataLoader) or not isinstance(
                test_loader, DataLoader):
            raise TypeError("train_loader, val_loader, and test_loader must be DataLoader instances.")

        self.model.to(self.device)

        scheduler = self._create_scheduler(len(train_loader))
        self.logger.info('Training process started.')

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            total_train_loss = 0
            correct_train_preds = 0
            total_train_preds = 0
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'Epoch {epoch + 1}/{self.epochs}, Current LR: {current_lr}')

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)

            for batch in progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device).long()  # CrossEntropyLoss expects integer class ids

                self.optimizer.zero_grad()
                logits = self.model(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'])  # (batch, num_labels)
                loss = self.loss_fn(logits, labels)

                loss_value = loss.item()
                total_train_loss += loss_value
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()  # Per-batch step (the schedule is defined in steps)

                # Predicted class = the one with the highest logit
                predictions = logits.argmax(dim=-1)
                correct_train_preds += (predictions == labels).sum().item()
                total_train_preds += labels.size(0)

                train_accuracy = correct_train_preds / total_train_preds if total_train_preds > 0 else 0.0
                progress_bar.set_postfix(loss=loss_value, accuracy=train_accuracy)

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = correct_train_preds / max(total_train_preds, 1)
            self.logger.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Train Acc: {train_accuracy}")

            # Validation step after each epoch
            avg_val_loss, val_accuracy = self.evaluate(val_loader, 'validation')

            # Record metrics BEFORE the early-stopping check, so the last epoch is not lost
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            final_train_accuracy = train_accuracy

            # Checkpoint if this is the best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                self.save_best_model()
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= self.patience:
                self.early_stopping_triggered = True
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        self.logger.info('Training process completed. Starting testing on the test set.')

        # Load the best model before testing
        self.load_best_model()

        avg_test_loss, final_test_accuracy = self.evaluate(test_loader, 'testing')
        self.logger.info(f"Test Loss: {avg_test_loss}, Test Accuracy: {final_test_accuracy}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_train_accuracy, final_test_accuracy

    def evaluate(self, data_loader, context='validation'):
        assert context in ['validation', 'testing'], "Context must be either 'validation' or 'testing'"

        self.model.eval()
        self.model.to(self.device)
        total_loss = 0
        y_true, y_pred, y_prob = [], [], []  # y_prob: (N, num_labels) softmax probabilities

        if not hasattr(self, f'evaluation_accuracies_{context}'):
            setattr(self, f'evaluation_accuracies_{context}', [])

        progress_bar = tqdm(data_loader, desc=f"Evaluating ({context})", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device).long()

                logits = self.model(input_ids=inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'])  # (batch, num_labels)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                probabilities = torch.softmax(logits, dim=-1)
                predictions = probabilities.argmax(dim=-1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_prob.append(probabilities.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.concatenate(y_prob, axis=0)
        label_ids = list(range(self.num_labels))

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(y_true, y_pred)

        # Multi-class metrics: macro = unweighted mean over classes, weighted = weighted by class support
        precision_macro = precision_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, labels=label_ids, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=label_ids, average='weighted', zero_division=0)

        # One-vs-rest AUC; fails if some class is absent from y_true
        try:
            auc_roc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro', labels=label_ids)
        except ValueError as e:
            self.logger.warning(f"AUC-ROC could not be computed: {e}")
            auc_roc = float('nan')

        report = classification_report(y_true, y_pred, labels=label_ids, target_names=self.class_names,
                                       digits=4, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)

        self.logger.info(f"{context.capitalize()} Evaluation Metrics:")
        self.logger.info(f"  - Average Loss: {avg_loss}")
        self.logger.info(f"  - Accuracy: {accuracy}")
        self.logger.info(f"  - Macro Precision: {precision_macro}")
        self.logger.info(f"  - Macro Recall: {recall_macro}")
        self.logger.info(f"  - Macro F1: {f1_macro}")
        self.logger.info(f"  - Weighted F1: {f1_weighted}")
        self.logger.info(f"  - Macro AUC-ROC (OvR): {auc_roc}")
        self.logger.info(f"  - Per-class report:\n{report}")
        self.logger.info(f"  - Confusion Matrix (rows = actual, cols = predicted, order = {self.class_names}):\n{cm}")

        self.last_metrics[context] = {
            'loss': avg_loss, 'accuracy': accuracy, 'precision_macro': precision_macro,
            'recall_macro': recall_macro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
            'auc_roc_macro_ovr': auc_roc,
        }
        getattr(self, f'evaluation_accuracies_{context}').append(accuracy)

        # Validation plots are saved (overwritten each epoch) but never shown; test plots can be shown
        show = self.show_plots and context == 'testing'
        self.plot_absolute_confusion_matrix(y_true, y_pred, context, show=show)
        self.plot_normalized_confusion_matrix(y_true, y_pred, context, show=show)
        self.plot_roc_curve(y_true, y_prob, context, show=show)

        return avg_loss, accuracy

    # -----------------------------------------------------------------Utility Functions-------------------------------------------------------------
    """
    save_best_model is used during the training process, typically after each epoch, to overwrite the same file with the best model's state so far. This is saved in your checkpoint directory.
    save is used to save the final model state after training has finished, regardless of whether it's the best according to validation metrics. This is saved in your final_model directory.
    """
    def save_best_model(self):
        model_save_path = os.path.join(self.checkpoint_path, "best_model.pt")
        torch.save(self.model.state_dict(), model_save_path)
        self.logger.info(f"Best model saved to {model_save_path}")

    def save(self, file_path):
        """Save the model state dictionary to the specified file path."""
        try:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            torch.save(self.model.state_dict(), file_path)
            self.logger.info(f"Model state dictionary saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def plot_metrics(self):
        """Plots the training/validation losses and accuracies per epoch."""
        self.logger.info('Plotting metrics.')
        if not self.train_losses or not self.val_losses:
            self.logger.info('No training/validation data available to plot.')
            return

        epochs_range = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.train_losses, label='Training Loss', marker='x')
        plt.plot(epochs_range, self.val_losses, label='Validation Loss', marker='o')
        plt.title('Loss over time')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.train_accuracies, label='Training Accuracy', marker='x')
        plt.plot(epochs_range, self.val_accuracies, label='Validation Accuracy', marker='o')
        plt.title('Accuracy over time')
        plt.xlabel('Epoch')
        plt.legend(loc='best')

        plt.tight_layout()
        finish_figure(os.path.join(self.results_path, 'training_metrics.png'), self.show_plots)

    def plot_roc_curve(self, y_true, y_prob, context='validation', show=False):
        """One-vs-rest ROC curve for each class."""
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_labels)))  # (N, num_labels)

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
        plt.title(f'{context.capitalize()} ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.grid(True)
        finish_figure(os.path.join(self.results_path, f'roc_curve_{context}.png'), show)

    def _plot_confusion_matrix(self, cm, context, normalized, show=False):
        values = cm
        if normalized:
            row_sums = cm.sum(axis=1, keepdims=True)
            values = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

        title = f'{context.capitalize()} Confusion Matrix' + (' (Normalized)' if normalized else '')
        cmap = plt.cm.Reds if normalized else plt.cm.Blues
        value_format = '.2f' if normalized else 'd'

        plt.figure(figsize=(7, 6))
        plt.imshow(values, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(self.num_labels)
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')

        thresh = values.max() / 2.
        for i, j in itertools.product(range(values.shape[0]), range(values.shape[1])):
            plt.text(j, i, format(values[i, j], value_format),
                     horizontalalignment="center",
                     color="white" if values[i, j] > thresh else "black")

        plt.tight_layout()
        suffix = '_normalized' if normalized else ''
        finish_figure(os.path.join(self.results_path, f'confusion_matrix_{context}{suffix}.png'), show)

    def plot_absolute_confusion_matrix(self, y_true, y_pred, context='validation', show=False):
        assert context in ['validation', 'testing'], "Context must be either 'validation' or 'testing'"
        self.logger.info(f'Plotting absolute confusion matrix for {context}.')
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_labels)))
        self._plot_confusion_matrix(cm, context, normalized=False, show=show)

    def plot_normalized_confusion_matrix(self, y_true, y_pred, context='validation', show=False):
        """Row-normalized: the diagonal shows each class's recall."""
        assert context in ['validation', 'testing'], "Context must be either 'validation' or 'testing'"
        self.logger.info(f'Plotting normalized confusion matrix for {context}.')
        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_labels)))
        self._plot_confusion_matrix(cm, context, normalized=True, show=show)

    def plot_final_accuracies(self, final_train_accuracy, final_test_accuracy):
        if final_train_accuracy is None or final_test_accuracy is None:
            print("Final training or testing accuracy not provided.")
            return

        plt.figure(figsize=(6, 4))
        plt.plot(['Train', 'Test'], [final_train_accuracy, final_test_accuracy], marker='o')
        for i, acc in enumerate([final_train_accuracy, final_test_accuracy]):
            plt.text(i, acc, f"{acc:.2f}", ha='center', va='bottom')

        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Final Training & Testing Accuracy')
        plt.legend(['Accuracy'])
        finish_figure(os.path.join(self.results_path, 'final_accuracies.png'), self.show_plots)