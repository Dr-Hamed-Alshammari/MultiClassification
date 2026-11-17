# Arabic Sentiment Classification (Negative / Neutral / Positive)

Fine-tunes **XLM-RoBERTa** (`xlm-roberta-base`) to classify Arabic text into three classes:
**negative**, **neutral** and **positive**. Arabic diacritics are removed automatically before tokenization.

## Project structure

```
MultiClassification/
├── config.json              # All hyperparameters and paths (edit this, not the code)
├── main.py                  # Train + validate + test
├── Evaluator.py             # Evaluate a saved model on the test set / predict new texts
├── Classifier.py            # Model, training loop, metrics and plots
├── ClassifierDataset.py     # Reads the CSV files and tokenizes the text
├── ClassifierDataLoader.py  # Builds the train / validation / test DataLoaders
├── utils.py                 # Logging and plot-saving helpers
├── requirements.txt
├── Dataset/                 # <-- put your three CSV files here
└── sample_dataset/          # Tiny example data to check that your setup works
```

## 1. Setup

1. Install Python 3.10 or newer.
2. Install **PyTorch** using the command for your system from
   <https://pytorch.org/get-started/locally/> (choose the CUDA version if you have an NVIDIA GPU).
3. Install the other packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Check that PyTorch sees your GPU (it should print `True`):

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

   The code also runs on the CPU, but training will be very slow.

The first run downloads `xlm-roberta-base` (about 1 GB) from Hugging Face, so you need an internet connection.

## 2. Prepare the dataset

Put **three CSV files directly inside the `Dataset/` folder** (no subfolders):

```
Dataset/
├── Training.csv
├── Validation.csv
└── Testing.csv
```

Each file must have exactly these two columns:

| text                        | label    |
|-----------------------------|----------|
| الخدمة ممتازة والموظفون متعاونون | positive |
| وصل الطلب اليوم              | neutral  |
| المنتج سيء ولا أنصح به        | negative |

- `label` can be a word (`negative`, `neutral`, `positive`) or a number (`0`, `1`, `2`).
- Numbers follow the order of `class_names` in `config.json`: `0 = negative`, `1 = neutral`, `2 = positive`.
- Save the files as **UTF-8** CSV (in Excel: *Save As → CSV UTF-8*), otherwise Arabic text may be corrupted.
- Rows with an empty `text` or `label` are skipped; an unknown label stops the program with an error naming the row.

## 3. Quick test with the sample data (recommended first)

Before using your real data, check that everything works:

1. In `config.json`, set `"root_folder": "sample_dataset"` and `"epochs": 2`.
2. Run `python main.py`.
3. When it finishes, set `"root_folder"` back to `"Dataset"` and `"epochs"` back to `10`.

The sample has only 36 sentences, so the accuracy will be poor. The goal is only to check that the code runs.

## 4. Train

Run from the project folder (the folder that contains `config.json`):

```bash
python main.py
```

In PyCharm: open the project folder, right-click `main.py` → **Run**. Make sure the run configuration's
*Working directory* is the project folder.

What happens:

1. The model trains for up to `epochs` epochs and is evaluated on the validation set after each epoch.
2. Whenever the validation loss improves, the model is saved to `sentiment_checkpoints/best_model.pt`.
3. Training stops early if the validation loss does not improve for `patience` epochs.
4. The best model is evaluated on the test set.
5. The final model is saved to `final_model/Saved_Arabic_Sentiment_Model.pt`.

## 5. Evaluate a trained model / predict new texts

```bash
python Evaluator.py
```

This loads `sentiment_checkpoints/best_model.pt`, evaluates it on `Dataset/Testing.csv`, and saves
`results/test_predictions.csv` (the true label, predicted label and class probabilities for every test text).

To classify your own sentences, uncomment the example at the bottom of `Evaluator.py`:

```python
labels, probs = evaluator.predict(["الخدمة ممتازة جدا", "المنتج سيء"], tokenizer)
print(labels)   # e.g. ['positive', 'negative']
```

## 6. Outputs

| Where                     | What                                                                    |
|---------------------------|-------------------------------------------------------------------------|
| `classifier.log`          | One log file for **all** runs (new runs are appended), each starting with the full configuration |
| `results/`                | Confusion matrices, ROC curves, loss/accuracy curves, `test_predictions.csv` |
| `sentiment_checkpoints/`  | `best_model.pt` (best validation loss)                                  |
| `final_model/`            | Model saved at the end of training                                      |

Metrics reported: accuracy, macro precision / recall / F1, weighted F1, macro one-vs-rest AUC-ROC,
a per-class report and the confusion matrix.

## 7. Configuration (`config.json`)

| Key                     | Default             | Meaning |
|-------------------------|---------------------|---------|
| `model_name`            | `xlm-roberta-base`  | Any Hugging Face encoder model |
| `num_labels`            | `3`                 | Number of classes |
| `class_names`           | `negative, neutral, positive` | Class names; their order defines the label numbers |
| `use_class_weights`     | `false`             | `true` gives rare classes more weight in the loss (useful for imbalanced data) |
| `max_length`            | `256`               | Maximum tokens per text; longer texts are cut |
| `learning_rate`         | `3e-5`              | Peak learning rate |
| `initial_learning_rate` | `1e-9`              | Learning rate at the start of warmup |
| `warmup_epochs`         | `1`                 | Epochs of linear warmup, followed by cosine decay |
| `epochs`                | `10`                | Maximum number of epochs |
| `batch_size`            | `16`                | Training and validation batch size |
| `testing_batch_size`    | `32`                | Test batch size |
| `num_workers`           | `2`                 | Background processes for loading data (set to `0` if you get DataLoader errors) |
| `patience`              | `5`                 | Early-stopping patience (epochs without validation-loss improvement) |
| `resume`                | `false`             | `true` continues from `best_model.pt`; `false` starts from scratch and overwrites it |
| `show_plots`            | `false`             | `true` pops up the final plots; plots are always saved in `results/` |
| `root_folder`           | `Dataset`           | Folder containing the three CSV files |
| `seed`                  | `1`                 | Random seed for reproducibility |

## Troubleshooting

- **`CUDA out of memory`**: lower `batch_size` (e.g. 8) and/or `max_length` (e.g. 128).
- **`Dataset does not exist`**: run the script from the project folder, and check `root_folder` in `config.json`.
- **`unknown label`**: a label in your CSV is misspelled or isn't one of `class_names`.
- **`Checkpoint ... does not match the current model`**: `best_model.pt` came from a different model
  (e.g. the old binary version or a different `num_labels`). Delete it, or set `resume` to `false`.
- **Errors mentioning DataLoader workers (often on Windows)**: set `"num_workers": 0`.
