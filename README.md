# Deshift

Deshift is a library for distributionally robust optimization in PyTorch.

## Installation

Deshift requires PyTorch >= 1.6.0. Please go [here](https://pytorch.org/) for instructions 
on how to install PyTorch based on your platform and hardware.
Once Pytorch is installed you can install Deshift by running the following on the command line from the
root folder:
```
pip install -e .
```

Additional dependencies to run the example in `examples/train_fashion_mnist.ipynb` can be installed using `pip install -e .[examples]`. To build the docs, additional dependencies can be run using `pip install -e .[docs]`.

## Quickstart

A detailed quickstart example is present in the docs `docs/source/quickstart.ipynb`. At a glance, the functionality is a follows. First, we construct a function that inputs a vector of losses and returns a probability distribution over elements in this loss vector.
```
>>> from deshift import make_spectral_risk_measure, make_superquantile_spectrum
>>> spectrum = make_superquantile_spectrum(batch_size, 0.5)
>>> compute_sample_weight = make_spectral_risk_measure(spectrum, penalty="chi2", shift_cost=1.0)
```
Assume that we have computed a vector of losses based on a model output in PyTorch. We can then use the function above and back propagate through the weighted sum of losses.
```
>>> x, y = get_batch()
>>> logits = model(x)
>>> losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
>>> weights = compute_sample_weight(losses)
>>> loss = weights @ losses
>>> loss.backward()
```

## Documentation

The documentation is available [here](https://ronakdm.github.io/deshift/).

## Contributing

If you find any bugs, please raise an issue on GitHub.
If you would like to contribute, please submit a pull request.
We encourage and highly value community contributions.

## Citation

If you find this package useful, or you use it in your research, please cite:

    @inproceedings{mehta2023stochastic,
      title={{Stochastic Optimization for Spectral Risk Measures}},
      author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Liu, Lang and Harchaoui, Zaid},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      pages={10112--10159},
      year={2023},
      organization={PMLR}
    }

## Acknowledgments

This work was supported by NSF DMS-2023166, CCF-2019844, DMS-2134012, NIH, and the Office of the Director of National Intelligence (ODNI)’s IARPA program via 2022-22072200003. Part of this work was done while Zaid Harchaoui was visiting the Simons Institute for the Theory of Computing. The views and conclusions contained herein are those of the authors and should not be interpreted as representing the official views of ODNI, IARPA, or the U.S. Government.




