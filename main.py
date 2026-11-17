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
from utils import setup_logging, log_run_header


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # Load configuration (run this script from the project folder, next to config.json)
    with open('config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)

    # One appended log file for all runs + console output
    setup_logging(config.get('log_file', 'classifier.log'))
    logger = logging.getLogger(__name__)
    log_run_header(logger, 'training', config)

    seed = config['seed']
    set_seed(seed)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no CUDA GPU found)'
    logger.info(f"Device: {device_name}")
    logger.info(f"Current Working Directory: {os.getcwd()}")

    # Set up full paths for saving models, checkpoints and results
    project_root_dir = os.getcwd()
    model_save_path = os.path.join(project_root_dir, config['model_save_path'])
    final_model_path = os.path.join(project_root_dir, config['final_model_path'])
    model_checkpoints = os.path.join(project_root_dir, config['checkpoint_path'])
    results_path = os.path.join(project_root_dir, config.get('results_path', 'results'))
    for directory in (model_save_path, final_model_path, model_checkpoints, results_path):
        os.makedirs(directory, exist_ok=True)

    batch_size = config['batch_size']
    test_batch_size = config.get('testing_batch_size', batch_size)
    num_labels = config['num_labels']
    class_names = config['class_names']

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
        num_workers=config.get('num_workers', 2),
        pin_memory=torch.cuda.is_available()
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
        class_weights=class_weights,
        show_plots=config.get('show_plots', False),
        results_path=results_path
    )

    # Resume only when explicitly requested in config.json
    best_model_path = os.path.join(model_checkpoints, "best_model.pt")
    if config.get('resume', False):
        if classifier.load_best_model():
            logger.info("resume=true: continuing training from the existing best model.")
        else:
            logger.info("resume=true but no compatible checkpoint was found. Starting from scratch.")
    elif os.path.isfile(best_model_path):
        logger.info("resume=false: starting from scratch. The existing best_model.pt will be overwritten.")
    else:
        logger.info("Starting training from scratch.")

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

        logger.info(f"Final test metrics: {classifier.last_metrics.get('testing')}")
        logger.info(f"Plots saved in: {results_path}")
        logger.info("TRAINING RUN FINISHED")

    except Exception:
        logger.exception("An error occurred")  # Logs the full traceback to the console and the log file
        raise


if __name__ == '__main__':
    main()
