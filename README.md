# BART Model for the Rheological Property Prediction from LMWGs

This repository contains the dataset and scripts used to train BART models for the prediction of low molecular weight gelator rheological properties. The notebooks folder contains an example notebook for the prediction of rheological properties from smiles strings.

For ease of use, we also provide a [Google Colab implementation](https://colab.research.google.com/github/cmwoodley/BART_LMWG_model/blob/master/notebooks/BART_LMWG.ipynb) of our code to predict rheological properties in a web browser.

## Requirements
- pymc3
- arviz
- sklearn
- rdkit
- matplotlib
- seaborn

## Installation

Installation should take approximately 10 minutes on a normal PC.

1. Clone this repository:
```
git clone https://github.com/cmwoodley/BART_LMWG_model.git
```
2. Create conda environment and install the required packages:
```
conda create -n BART_LMWG python=3.8 pymc3==3.11.5 arviz rdkit matplotlib numpy=1.20 numba=0.56 pandas dill==0.3.5.1 seaborn scikit-learn ipykernel -c conda-forge
```

## Usage

To build these models locally, run the training script provided in scripts/train.py:
```
python train.py
```

Building models with train.py should take less than 10 minutes on a normal PC.

Serialised models are saved in ./models. Summary of predictions and scoring metrics are saved in the reports folder.

An example notebook (notebooks/notebook1.ipynb) is provided with examples of predictions on a simgle LMWG and batches of LMWG.

## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Contact
If you have any questions or suggestions, please feel free to contact me at cwoodley@liverpool.ac.uk.
