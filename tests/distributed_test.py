"""Tests for distributed extensions."""

from absl.testing import absltest

import numpy as np
import torch
import torch.distributed as dist

import sys
sys.path.append(".")
import deshift._src.spectral_risk as spectral_risk
import deshift._src.spectra as spectra
from deshift._src.distributed import ddp_max_oracle

dist.init_process_group(backend="nccl")

class DistributedTest(absltest.TestCase):

  def test_spectral_risk(self):
    micro_sizes = torch.tensor([10, 5, 15])
    batch_size = micro_sizes.sum()
    spectrum = spectra.make_extremile_spectrum(batch_size, 2.0)
    shift_cost = 1.0
    penalty = "chi2"
    max_oracle = spectral_risk.make_spectral_risk_measure(spectrum, penalty, shift_cost)

    np.random.seed(123)
    losses = np.random.normal(size=(batch_size))
    weights1 = max_oracle(losses).tolist()
    weights2 = ddp_max_oracle(max_oracle, losses, micro_sizes).cpu().tolist()
    self.assertSequenceAlmostEqual(weights1, weights2, places=6)

if __name__ == '__main__':
  absltest.main()

dist.destroy_process_group()