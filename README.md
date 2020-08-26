# Optimization of cut-offs for repeater chains
This repository stores the implementation of the algorithm introduced in *Efficient optimization of cut-offs in quantum repeater chains* by Boxi Li, Tim Coopmans and David Elkouss. It includes two implementations: 
- The numerical algorithm calculating the waiting time distribution and the fidelity of the delivered entangled state.
- The optimizer used to optimize the cut-off time for maximal secret key rate.

## Tutorial [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/BoxiLi/repeater-cut-off-optimization/master?filepath=https%3A%2F%2Fgithub.com%2FBoxiLi%2Frepeater-cut-off-optimization%2Fmaster%2FExample.ipynb)
A tutorial is written in Jupyter Notebook. By clicking the badge above, you will be directed to a online Jupyter Notebook. After the loading succeed, please find and click `Example.ipynb`. Then you can run examples in the notebook online without installing anything! (Loading could be a few minutes, please be patient.)


## Download
To download or clone the repository, using the green button `Clone or download`.

## Prerequisites
The following Python packages are required for running the core algorithms:
```
NumPy, Scipy, Numba
```
In addition, we use `Matplotlib` for plotting and `pytest` for unit tests.

To install the packages, you can use
```
pip install numpy scipy numba matplotlib pytest
```
or 
```
conda install numpy scipy numba matplotlib pytest
```
if you are using [conda](https://docs.conda.io/en/latest/) environment.

For GPU accelerated convolution, you will need
```
CuPy
```
See [CuPy installation](https://docs-cupy.chainer.org/en/stable/install.html) for details

## File overview
- The protocol units such as entanglement swap, distillation or cut-off are defined in `protocol_units.py`.
- The core code for the numerical simulation of repeater chains is under `repeater_algorithm.py`.
- The optimizer can be found in `optimize_cutoff.py`.
- Examples for computing repeater protocols and optimizing the cut-off time are given in `examples.py`
- The figures in the paper can be reproduced by the code stored in `plot_paper.py`. One can either use the prepared data (saved in the directory `data`) or produce those data anew.
