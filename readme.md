# An Analysis of Fine-Tuning Methods for Image Captioning Models on Downstream Tasks

![A sample from the CUB-200 Dataset](readme_image.png)

The repository used for conducting the experiments for the paper "An Analysis of Fine-Tuning Methods for Image Captioning Models on Downstream Tasks" by Victor Coelho de Souza and Daniel Silva Ferreira.

This repository allows the comparison of different fine-tuning methodologies for Image Captioning Models on a given dataset. On the paper, the performance of Full Fine-Tuning, LoRA, LayerNorm Tuning, Wise-FT and Bottleneck Adapters were compared using the datasets CUB-200 and RSICD. Additional information can be found on the research paper.

### Authors

Authors: Victor Coelho de Souza and Daniel Silva Ferreira

Institution: Ceará Federal Institute of Technology

### Installation

The Anaconda package manager is necessary to install the dependencies used in this project. The dependencies are provided in the file ```conda_env.yml``` and can be installed with the following command:

```conda env create -f conda_env.yml```

An Nvidia GPU with CUDA support will be needed for both training and testing the model.

### Usage

To train the Image Captioning model on a dataset, the script ```src/entrypoints/training.py``` is provided. You can execute it by running the following command:

```python src/entrypoints/training.py```

The entrypoint ```src/entrypoints/testing.py``` is also available to test fine-tuned models. Running it will output the result of evaluating the trained model with 4 key metrics: SPICE, CIDEr, BertScore and METEOR on the predefined testing subsets of the CUB-200 and RSICD datasets.

To use this repository yourself, editing both files (training.py and testing.py), so they can point to the correct directories and files, will be necessary, as well as providing your own datasets. The code provided has also not been tested with other datasets.
