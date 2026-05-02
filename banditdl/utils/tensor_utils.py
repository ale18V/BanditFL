"""Tensor utility functions for flattening and unflattening model parameters."""

import torch


def flatten(list_of_tensors):
    """Flatten list of tensors. Used for model parameters and gradients."""
    return torch.cat(tuple(tensor.view(-1) for tensor in list_of_tensors))


def unflatten(flat_tensor, model_shapes):
    """Unflatten a flat tensor. Used when setting model parameters and gradients."""
    c = 0
    returned_list = [torch.zeros(shape) for shape in model_shapes]
    for i, shape in enumerate(model_shapes):
        count = 1
        for element in shape:
            count *= element
        returned_list[i].data = flat_tensor[c:c + count].view(shape)
        c = c + count
    return returned_list
