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
                
    #     # last_sample += samples_for_worker
    #     # Integer truncation can leave samples unassigned; distribute remainder round-robin.
    #     remainder = indices_per_label[label][last_sample:]
    #     for offset, idx in enumerate(remainder):
    #         worker_samples[offset % nb_workers].append(idx)

    # # With many workers and skewed proportions, some workers can still end up with no indices
    # # (e.g. int proportions and remainder never land on their index). Steal one sample at a time
    # # from a worker that can spare it until no empty worker remains, when possible.
    # while True:
    #     empty_workers = [w for w in range(nb_workers) if len(worker_samples[w]) == 0]
    #     if not empty_workers:
    #         break
    #     donors = [w for w in range(nb_workers) if len(worker_samples[w]) > 1]
    #     if not donors:
    #         break
    #     donor = max(donors, key=lambda w: len(worker_samples[w]))
    #     moved = worker_samples[donor].pop()
    #     worker_samples[empty_workers[0]].append(moved)

    return worker_samples
