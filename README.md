# UMU-Ev at SatiSPeech 2025: Multimodal Satire Detection in Spanish

This repository contains the full codebase for the UMU-Ev system developed for
the SatiSPeech 2025 shared task (IberLEF). The task focused on automatic satire
detection in Spanish using multimodal signals, specifically text and audio. The
proposed system explores a range of efficient representations and fusion
strategies, aiming to achieve competitive performance under limited
computational resources.

## Key components

- **Embedding Generation Scripts** (`embed_scripts`):
Python scripts for extracting and saving text and audio embeddings in `.npy`
format. These scripts process input data from `.csv` files and apply models
such as `RoBERTa`, `FastText`, `HuBERT`, and `MFCC`-based features.

- **Model Training and Evaluation** (`train_scripts`):
Scripts to train classification models (_SVM_, _DNN_, _Logistic Regression_ and
_MLP_) using precomputed embeddings. Results are saved in structured `.json`
files, including performance metrics and hyperparameters.

- **Prediction and Output Handling**:
Classification reports for the models are generated during the training process
and stored in `.json` files inside the `results` folder. These files are meant
to be used for evaluation and analysis. Predictions are made using the trained
models and the predictions are saved in `.csv`.

## Instalation

The project contains a `requirements.txt` file with the required packages. To
install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

It also contains a `pyproject.toml` file with the project configuration in order to
use a package manager such as `poetry` or `uv`.

### Files Required for Embedding Generation

Files needed for embedding generation are stored in the `embedding_files` folder.

FastText:

- `cc.es.300.bin`: FastText model file.

Word2Vec (conversion to `.npy` vectors was needed due to the limitation of the system):

- `word2vec_vectors.npy`: Word2Vec vectors file.
- `word2vec_vocab.npy`: Word2Vec vocabulary file.

## Notes

The present attention mechanism is a placeholder made so it doesn't break the program if used.
It uses a simple attention mechanism that assigns random weights to all tokens and doesn't train
them in the training the process.

The attention mechanism used in the project is still to be ported from the original notebook
implementation to the scripts presented in this repository.
