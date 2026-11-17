import os
import json
import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from Classifier import Classifier
from ClassifierDataset import ClassifierDataset
from ClassifierDataLoader import ClassifierDataLoader


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Load configuration
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Set seed for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info("Current Working Directory: {}".format(os.getcwd()))

    # Set up full paths for saving models and checkpoints
    project_root_dir = os.getcwd()
    model_save_path = os.path.join(project_root_dir, config['model_save_path'])
    final_model_path = os.path.join(project_root_dir, config['final_model_path'])
    model_checkpoints = os.path.join(project_root_dir, config['checkpoint_path'])
    batch_size = config['batch_size']
    test_batch_size = config.get('testing_batch_size', batch_size)
    num_labels = config['num_labels']
    class_names = config['class_names']

    # Ensure that directories exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(final_model_path, exist_ok=True)
    os.makedirs(model_checkpoints, exist_ok=True)

    # Load tokenizer
    model_name = config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize datasets
    train_dataset = ClassifierDataset(tokenizer, 'train')
    val_dataset = ClassifierDataset(tokenizer, 'val')
    test_dataset = ClassifierDataset(tokenizer, 'test')

    # Optional class weights (inverse frequency) for imbalanced sentiment data
    class_weights = None
    if config.get('use_class_weights', False):
        counts = train_dataset.get_class_counts().float()
        class_weights = counts.sum() / (num_labels * counts.clamp(min=1))
        logger.info(f"Using class weights: {dict(zip(class_names, class_weights.tolist()))}")

    # Initialize data loaders
    data_loader = ClassifierDataLoader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        seed=seed,
        num_workers=4,
        pin_memory=True
    )
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()

    # Initialize the classifier
    classifier = Classifier(
        model_name=model_name,
        num_labels=num_labels,
        class_names=class_names,
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        checkpoint_path=model_checkpoints,
        patience=config['patience'],
        initial_learning_rate=config['initial_learning_rate'],
        warmup_epochs=config['warmup_epochs'],
        class_weights=class_weights
    )

    # Load the best model if a compatible one exists, else start training from scratch
    best_model_path = os.path.join(model_checkpoints, "best_model.pt")
    if classifier.load_best_model():
        logger.info("Loaded an existing best model; training will continue from it.")
    else:
        logger.info("No compatible best model found. Starting training from scratch.")

    classifier.to(classifier.device)

    # Train and evaluate
    try:
        final_train_accuracy, final_test_accuracy = classifier.train(train_loader, val_loader, test_loader)

        classifier.plot_metrics()
        classifier.plot_final_accuracies(final_train_accuracy, final_test_accuracy)

        # Save final model
        final_model_save_path = os.path.join(final_model_path, "Saved_Arabic_Sentiment_Model.pt")
        logger.info(f"Attempting to save the final model to: {final_model_save_path}")
        classifier.save(final_model_save_path)

        if os.path.isfile(best_model_path):
            logger.info("The best model was saved successfully.")
        else:
            logger.info("The best model was not saved.")

        logger.info(f"Test metrics: {classifier.last_metrics.get('testing')}")

    except Exception:
        logger.exception("An error occurred")  # Logs the full traceback


if __name__ == '__main__':
    main()