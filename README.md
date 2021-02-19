# Introduction

This repo contains code for reproducing the results in the paper [**Query Training: Learning a Worse Model to Infer Better Marginals in Undirected Graphical Models with Hidden Variables**](https://arxiv.org/abs/2006.06803) at the 35th AAAI Conference on Artificial Intelligence (AAAI 2021).

# Setting up the environment

1. [Install miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) (if not already installed)
2. Set up the virtual environment
```
conda env create -f environment.yml
conda activate qt
python setup.py develop
```
3. Download the [data](http://vcrs-public-aaai-2021-query-training.s3-website-us-west-2.amazonaws.com/) and point `BASE` in `query_training/__init__.py` to the data directory

The code was tested on Ubuntu 18.04 with CUDA 10.1.

# Reproducing the results

Use the scripts in the `scripts` folder to reproduce results in the paper.
