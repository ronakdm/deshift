"""Functions relating to spectral risk measure ambiguity sets."""
from functools import partial

import torch
import numpy as np

from deshift._src import pav

def make_spectral_risk_measure(
        spectrum: np.ndarray,
        penalty: str="chi2",
        shift_cost: float=0.0,
    ):
    """Create a function which computes the sample weights from a vector of losses when using a spectral risk measure ambiguity set.
 
    Args:
      spectrum: a Numpy array containing the spectrum weights, 
        which should be the same length as the batch size.
      penalty: either 'chi2' or 'kl' indicating which f-divergence 
        to use as the dual regularizer.
      shift_cost: the non-negative dual regularization parameter.
      group_dist

    Returns:
      compute_sample_weight
        a function that maps ``n`` losses to a vector of ``n`` weights on each training example.
    """
    def max_oracle(losses):
        assert torch.is_tensor(losses), "`losses` must be a PyTorch tensor"
        with torch.no_grad():
            device = losses.get_device()
            device = device if device >= 0 else "cpu"
            weights = spectral_risk_measure_maximization_oracle(spectrum, shift_cost, penalty, losses.cpu().numpy())
        return weights.to(device)
    return max_oracle

def make_group_spectral_risk_measure(
        spectrum: np.ndarray,
        penalty: str="chi2",
        shift_cost: float=0.0,
    ):
    """Create a function which computes the sample weights for Group DRO from a vector of losses when using a spectral risk measure ambiguity set.
 
    Args:
      spectrum: a Numpy array containing the spectrum weights, 
        which should be the same length as the number of groups.
      penalty: either 'chi2' or 'kl' indicating which f-divergence 
        to use as the dual regularizer.

    Returns:
      compute_sample_weight
        a function that maps ``n`` losses to a vector of ``n`` weights on each training example.
    """
    def max_oracle(losses, group_labels):
        assert torch.is_tensor(losses), "`losses` must be a PyTorch tensor"
        with torch.no_grad():
            device = losses.get_device()
            device = device if device >= 0 else "cpu"

            # count average loss of each group
            unique_labels, labels_count = group_labels.unique(dim=0, return_counts=True)
            res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, group_labels, losses)
            res = res / labels_count.float().unsqueeze(1)
            group_weights = spectral_risk_measure_maximization_oracle(spectrum, shift_cost, penalty, res.cpu().numpy())

            # renormalize weight for groups that were observed
            group_weights[res<=1e-10] = 0.0
            group_weights /= (group_weights.sum() * labels_count)
            weights = torch.gather(group_weights, 0, group_labels)
        return weights.to(device)
    return max_oracle

def spectral_risk_measure_maximization_oracle(
        spectrum: np.ndarray, 
        shift_cost: float,
        penalty: str,
        losses: np.ndarray
    ):
    """Maximization oracle to compute the sample weights based on a particular spectral risk measure objective.

    Args:
      spectrum: a Numpy array containing the spectrum weights, 
        which should be the same length as the batch size.
      shift_cost: a non-negative dual regularization parameter.
      penalty: either ``chi2`` or ``kl`` indicating which f-divergence
        to use as the dual regularizer.
      losses: a Numpy array containing the loss incurred by the model
        on each example in the batch.

    Returns:
      sample_weight
        a vector of ``n`` weights on each training example.
    """
    if shift_cost < 1e-12:
        return torch.from_numpy(spectrum[np.argsort(np.argsort(losses))])
    sample_size = len(losses)
    scaled_losses = losses / shift_cost
    perm = np.argsort(losses)
    sorted_losses = scaled_losses[perm]

    if penalty == "chi2":
        primal_sol = pav.l2_centered_isotonic_regression(
            sorted_losses, spectrum
        )
    elif penalty == "kl":
        primal_sol = pav.neg_entropy_centered_isotonic_regression(sorted_losses, spectrum)
    else:
        raise NotImplementedError
    inv_perm = np.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    if penalty == "chi2":
        q = scaled_losses - primal_sol + 1 / sample_size
    elif penalty == "kl":
        q = np.exp(scaled_losses - primal_sol) / sample_size
    else:
        raise NotImplementedError
    return torch.from_numpy(q).float()
