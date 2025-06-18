from argparse import ArgumentParser
from dataclasses import dataclass
from enum import StrEnum, auto
import json
from pathlib import Path
from typing import Any, Optional
from numpy import ndarray, load
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from fusion_utils import fusion_concat, load_embeddings_npy
from classif_utils import get_classifiers, timeit
from keras_utils import build_model, get_tuner, early_stop

results = {}
predictions = {}


class Embedding(StrEnum):
    FASTTEXT = auto()
    MFCC_FULL = "mfcc-full"
    MFCC_PROSODIC = "mfcc-prosodic"
    MFCC_STATS = "mfcc-stats"
    WORD2VEC = auto()
    ROBERTA = auto()
    HUBERT_CLS = "hubert-cls"
    HUBERT_MEAN = "hubert-mean"
    W2V2_CLS = "w2v2-cls"
    W2V2_MEAN = "w2v2-mean"

    def type(self) -> str:
        if self in [
            Embedding.HUBERT_CLS,
            Embedding.HUBERT_MEAN,
            Embedding.W2V2_CLS,
            Embedding.W2V2_MEAN,
            Embedding.MFCC_STATS,
            Embedding.MFCC_PROSODIC,
            Embedding.MFCC_FULL,
        ]:
            return "audio"
        return "text"


@dataclass
class Args:
    embedding: Embedding
    additional: Optional[Embedding]
    train_size: int
    val_size: int
    random_state: int
    output_path: Path

    def __post_init__(self):
        """
        Sets embeddings names to lowercase, creates output directory with subdirectory by kind of embedding used
        and sets the name if none was given.

        Raises a ValueError if you try to combine two embeddings of different type.
        """
        self.output_path = self.output_path / self.embedding.type()
        self.output_path.mkdir(parents=True, exist_ok=True)
        if self.additional:
            if self.embedding.type() != self.additional.type():
                raise ValueError(
                    "combining two kinds of embeddings is not supported"
                )
        self.name = self.embedding.value
        if self.additional:
            self.name += "_" + self.additional.value
        self.output_path = self.output_path / self.name


def parse_args() -> Args:
    parser = ArgumentParser(
        description="Script that trains several models using the provided embeddings data as input and produces several results."
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=Embedding,
        choices=Embedding,
        required=True,
        help='Name of the embedding to use. The embedding must be previously created in the "embeddings" folder, in ".npy" format for both the test and train sets, along with the data CSV files. The name format should be "test_<embedding>.npy."',
    )
    parser.add_argument(
        "--additional",
        "-a",
        type=Embedding,
        choices=Embedding,
        required=False,
        help="Name of the additional embedding to use along with the main one. The format is the same as for the main embedding. If not present, only the main embedding is used.",
    )
    parser.add_argument(
        "--train-size",
        "-t",
        type=int,
        default=5500,
        required=False,
        help="Samples used for training. Default: 5500.",
    )
    parser.add_argument(
        "--val_size",
        "-v",
        type=int,
        default=500,
        required=False,
        help="Samples used for validation. Default: 500.",
    )
    parser.add_argument(
        "--random-state",
        "-r",
        type=int,
        default=420,
        required=False,
        help="Random state for experiments reproducibility. Default: 420.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=False,
        default=Path("results"),
        help="Output directory for the results JSON files (created if it does not exist). Default: ./results.",
    )
    args = parser.parse_args()
    return Args(
        embedding=args.embedding,
        additional=args.additional,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state,
        output_path=args.output,
    )


def load_dfs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the train and test CSV files from the data directory.
    """
    train_df = pd.read_csv(data_dir / "SatiSPeech_phase_2_train_public.csv")
    test_df = pd.read_csv(data_dir / "SatiSPeech_phase_2_test_public.csv")
    return train_df, test_df


def get_splits_idx(
    train_df: pd.DataFrame, train_size: int, val_size: int, random_state: int
) -> tuple[ndarray, ndarray]:
    """
    Gets the train and validation indices from the train dataframe given the train size and validation size.
    """
    train_idx, val_idx = train_test_split(
        train_df.index.values,
        train_size=train_size,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["label"],
    )
    return train_idx, val_idx  # type: ignore


def get_labels(
    train_df: pd.DataFrame, train_idx: ndarray, val_idx: ndarray
) -> tuple[ndarray, ndarray]:
    """Gets the train and validation labels from the train dataframe given the train and validation indices."""
    y_train = (
        train_df.loc[train_idx, "label"]
        .map({"satire": 1, "no-satire": 0})
        .values
    )
    y_val = (
        train_df.loc[val_idx, "label"].map({"satire": 1, "no-satire": 0}).values
    )
    return y_train, y_val  # type: ignore


def load_embeddings(
    data_dir: Path,
    train_idx: ndarray,
    val_idx: ndarray,
    embedding: str,
    additional: Optional[str],
) -> tuple[ndarray, ndarray, ndarray]:
    """Loads the train, validation and test embeddings from the data directory."""
    test_path = data_dir / f"embeddings/test_{embedding}.npy"
    train_path = data_dir / f"embeddings/train_{embedding}.npy"
    train, val, scaler = load_embeddings_npy(
        train_path,
        idx_train=train_idx,
        idx_val=val_idx,
    )
    assert scaler is not None
    test = scaler.transform(load(test_path))
    if additional:
        test_path = data_dir / f"embeddings/test_{additional.lower()}.npy"
        train_path = data_dir / f"embeddings/train_{additional.lower()}.npy"
        train_a, val_a, scaler_a = load_embeddings_npy(
            train_path, idx_train=train_idx, idx_val=val_idx
        )
        assert scaler_a is not None
        test_a = scaler_a.transform(load(test_path))
        train = fusion_concat(train, train_a)
        val = fusion_concat(val, val_a)
        test = fusion_concat(test, test_a)  # type: ignore
    return train, val, test  # type: ignore


@timeit("DNN", results)
def train_keras(
    X_train: ndarray,
    y_train: ndarray,
    X_val: ndarray,
    y_val: ndarray,
    X_test: ndarray,
    name: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Finds hyperparameters for a Keras DNN model and trains the best model on the train and validation sets.
    Returns the results and predictions for the test set.
    """
    print("\nTuning and fitting DNN: ")
    keras_builder = build_model(X_train, y_train)
    tuner = get_tuner(keras_builder, name, random_state)
    tuner.search(
        X_train,
        y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[early_stop],
    )
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = keras_builder(best_hps)
    best_model.fit(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stop],
    )
    y_pred = best_model.predict(X_val)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test = best_model.predict(X_test)
    y_test_classes = y_test.argmax(axis=1)
    report = classification_report(
        y_val, y_pred_classes, digits=4, output_dict=True
    )
    assert isinstance(report, dict)
    results["DNN"] = report
    results["DNN"]["Hyperparameters"] = tuner.get_best_hyperparameters()[
        0
    ].values
    print(
        "#### Report for DNN:\n####",
        results,
    )
    return results, {"DNN": y_test_classes}


def train_classificators(
    X_train: ndarray,
    y_train: ndarray,
    X_val: ndarray,
    y_val: ndarray,
    X_test: ndarray,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Finds hyperparameters for several classification models and trains the best model on the train and validation sets.
    Returns the results and predictions for the test set.
    """
    classifiers = get_classifiers(random_state)
    for name, model in classifiers.items():
        timed_training = timeit(name, results)(train_classificator)
        timed_training(X_train, y_train, X_val, y_val, X_test, name, model)
    return results, predictions


def train_classificator(X_train, y_train, X_val, y_val, X_test, name, model):
    """Trains a classification model on the train and validation sets and returns the results and predictions for the test set."""
    print(f"\nTuning and fitting: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_test = model.predict(X_test)
    report = classification_report(y_val, y_pred, output_dict=True, digits=4)
    assert isinstance(report, dict)
    results[name] = report
    predictions[name] = y_test
    results[name]["Hyperparameters"] = model.best_params_
    print(f"Report for {name}:\n", results[name])


def save_results(
    df_test: pd.DataFrame,
    results: dict[str, Any],
    predictions: dict[str, Any],
    output: Path,
):
    """Saves the results to a JSON file and the predictions to a CSV file."""
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(results, f)
    df = pd.DataFrame.from_dict(predictions, orient="index").transpose()
    df = df.map(lambda x: "satire" if x == 1 else "no-satire")
    df["id"] = df_test["uid"].str.removesuffix(".mp3")
    df.to_csv(output.with_suffix(".csv"), index=False)


def main():
    args = parse_args()
    data_path = Path.cwd() / "data/public_data"
    embeddings = (
        args.embedding.value
        if args.additional is None
        else args.embedding.value + "+" + args.additional.value
    )
    print(f"Experimento {args.name}, usando embedding {embeddings}.")
    print(f"Random state: {args.random_state}")
    print(f"Directorio de salida: {args.output_path}")
    train_df, test_df = load_dfs(data_path)
    train_idx, val_idx = get_splits_idx(
        train_df, args.train_size, args.val_size, args.random_state
    )
    X_train, X_val, X_test = load_embeddings(
        data_path, train_idx, val_idx, args.embedding, args.additional
    )
    y_train, y_val = get_labels(train_df, train_idx, val_idx)
    keras_results, keras_preds = train_keras(
        X_train, y_train, X_val, y_val, X_test, args.name, args.random_state
    )
    class_results, class_preds = train_classificators(
        X_train, y_train, X_val, y_val, X_test, args.random_state
    )
    save_results(
        test_df,
        keras_results | class_results,
        keras_preds | class_preds,
        args.output_path,
    )


if __name__ == "__main__":
    main()
