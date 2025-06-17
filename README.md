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
