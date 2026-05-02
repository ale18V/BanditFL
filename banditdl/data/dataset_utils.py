"""Dataset utility functions."""

import pathlib
import numpy as np


def get_default_root():
    """Lazy-initialize and return the default dataset root directory path."""
    # Generate the default path
    default_root = pathlib.Path(__file__).parent.parent / "datasets" / "cache"
    # Create the path if it does not exist
    default_root.mkdir(parents=True, exist_ok=True)
    # Return the path
    return default_root


def draw_indices(samples_distribution, indices_per_label, nb_workers):
    """Return the indices of the training datapoints selected for each honest worker.
    
    Used in case of Dirichlet distribution.
    
    Args:
        samples_distribution: Distribution of samples per worker per label
        indices_per_label: Dictionary mapping labels to their indices
        nb_workers: Number of workers
    
    Returns:
        Dictionary mapping worker ID to list of sample indices
    """
    #JS: Initialize the dictionary of samples per worker. Should hold the indices of the samples each worker possesses
    worker_samples = dict()
    for worker in range(nb_workers):
        worker_samples[worker] = list()

    for label, label_distribution in enumerate(samples_distribution):
        last_sample = 0
        number_samples_label = len(indices_per_label[label])
        #JS: Iteratively split the number of samples of label into chunks according to the worker proportions, and assign each chunk to the corresponding worker
        for worker, worker_proportion in enumerate(label_distribution):
            samples_for_worker = int(worker_proportion * number_samples_label)
            worker_samples[worker].extend(indices_per_label[label][last_sample:last_sample+samples_for_worker])
            last_sample = samples_for_worker

    return worker_samples
