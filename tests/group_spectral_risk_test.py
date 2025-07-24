"""Tests for Group DRO extensions of spectral risk measures."""

from absl.testing import absltest, parameterized

import numpy as np
import torch

import sys
sys.path.append(".")
import deshift._src.spectral_risk as spectral_risk
import deshift._src.spectra as spectra


class GroupSpectralRiskTest(parameterized.TestCase):

  @parameterized.product(penalty=["chi2", "kl"])
  def test_spectral_risk(self, penalty):

    batch_size = 30

    # every element is in a group of its own
    torch.manual_seed(123)
    spectrum = spectra.make_extremile_spectrum(batch_size, 2.0)
    shift_cost = 1.0
    max_oracle = spectral_risk.make_group_spectral_risk_measure(spectrum, penalty, shift_cost)
    group_labels = torch.arange(batch_size)
    losses = torch.randn(batch_size)
    weights_group = [elem.item() for elem in max_oracle(losses, group_labels)]
    weights_indiv = [
        elem.item() for elem in spectral_risk.spectral_risk_measure_maximization_oracle(spectrum, 1.0, penalty, losses.numpy())
    ]
    self.assertSequenceAlmostEqual(weights_group, weights_indiv, places=6)

    # data is split in two groups
    torch.manual_seed(456)
    shift_cost = 0.0
    spectrum = spectra.make_superquantile_spectrum(2, 0.5)
    max_oracle = spectral_risk.make_group_spectral_risk_measure(spectrum, penalty, shift_cost)
    losses1 = torch.randn(batch_size // 2)
    losses2 = torch.randn(batch_size // 2) + 10
    losses = torch.cat([losses1, losses2])
    group_labels = torch.cat([torch.zeros(batch_size // 2).long(), torch.ones(batch_size // 2).long()])
    weights_group = max_oracle(losses, group_labels)
    self.assertAlmostEqual(weights_group[:batch_size // 2].sum(), 0.0, places=6)

if __name__ == '__main__':
  absltest.main()