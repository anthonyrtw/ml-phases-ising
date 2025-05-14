# Machine Learning Phases of Matter

In this repository, we reproduce results from the paper by Carrasquilla and Melko titled [Machine learning phases of matter](https://www.nature.com/articles/nphys4035).

Ferromagnetic Ising lattices are generated using the Metropolis algorithm for a variety of temperatures. A convolutional neural network is then trained directly on the spin configurations of the lattices and their corresponding phases.

## Structure

The project is organized as follows:

```bash
├── data/
│   ├── square/ # Square lattices
│   │   ├── L10/ # Data for lattice side length, 10
│   │   │   └── lattices.pt
│   │   │   └── magnetizations.pt
│   │   │   └── temperatures.pt
│   │   ├── L20/
│   │   ├── L30/
│   │   ├── L40/
│   │   └── L60/
│   ├── triangular/ # Triangular lattices
│   │   ├── L10/ 
│   │   │   └── lattices.pt
│   │   │   └── magnetizations.pt
│   │   │   └── temperatures.pt
│   │   ├── L20/
│   │   ├── L30/
│   │   ├── L40/
│   │   └── L60/
├── figures/
├── data_gen.cpp
├── machine_learning.ipynb
├── requirements.txt
└── README.md
```

`data_gen.cpp` is used to generate the datasets under the `data` directory.
Training on the datasets is performed in the `machine_learning.ipynb` notebook.

## Set-up

### Python

The following packages are required:

- Python 3.9+ 
- PyTorch 
- Matplotlib

The essential Python packages can be installed here:

``` 
pip install -r requirements.txt
```

### C++

The dataset is generated using a C++ script `data_gen.cpp`, which requires Libtorch, PyTorch's C++ library. You can follow the installation and set-up instructions for Libtorch from the [official documentation](https://pytorch.org/cppdocs/installing.html).
