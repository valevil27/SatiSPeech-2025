from argparse import ArgumentParser
from dataclasses import dataclass
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
valid_embeddings = {
    "text": [
        "fasttext",
        "mfcc_full",
        "mfcc_prosodic",
        "mfcc_stats",
        "word2vec",
        "roberta",
    ],
    "audio": ["hubert_cls", "hubert_mean", "w2v2_cls", "w2v2_mean"],
}


@dataclass
class Args:
    name: str
    embedding: str
    additional: Optional[str]
    train_size: int
    val_size: int
    random_state: int
    output_path: Path

    def __post_init__(self):
        self.embedding = self.embedding.lower()
        type_embedding = Args.get_embedding_type(self.embedding)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_path / f"{type_embedding}_{self.name}"
        if not self.additional:
            return
        self.additional = self.additional.lower()
        additional_type_embedding = Args.get_embedding_type(self.additional)
        if type_embedding != additional_type_embedding:
            raise ValueError(
                "combining two kinds of embeddings is not supported"
            )

    @staticmethod
    def get_embedding_type(embedding: str) -> str:
        for k, v in valid_embeddings.items():
            if embedding in v:
                return k
        raise ValueError("embedding not supported")


def parse_args() -> Args:
    parser = ArgumentParser(
        description="Script that trains several models using the provided embeddings data as input and produces several results."
    )
    parser.add_argument(
        "--name", "-n", type=str, required=True, help="Experiment name."
    )
    parser.add_argument(
        "--embedding",
        "-e",
        type=str,
        required=True,
        help='Name of the embedding to use. The embedding must be previously created in the "embeddings" folder, in ".npy" format for both the test and train sets, along with the data CSV files. The name format should be "test_<embedding>.npy."',
    )
    parser.add_argument(
        "--additional",
        "-a",
        type=str,
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
        name=args.name,
        embedding=args.embedding,
        additional=args.additional,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state,
        output_path=args.output,
    )


def load_dfs(data_path: Path):
    train_df = pd.read_csv(data_path / "SatiSPeech_phase_2_train_public.csv")
    test_df = pd.read_csv(data_path / "SatiSPeech_phase_2_test_public.csv")
    return train_df, test_df


def get_splits_idx(
    train_df: pd.DataFrame, train_size: int, val_size: int, random_state: int
) -> tuple[ndarray, ndarray]:
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
    data_path: Path,
    train_idx: ndarray,
    val_idx: ndarray,
    embedding: str,
    additional: Optional[str],
) -> tuple[ndarray, ndarray, ndarray]:
    test_path = data_path / f"embeddings/test_{embedding}.npy"
    train_path = data_path / f"embeddings/train_{embedding}.npy"
    train, val, scaler = load_embeddings_npy(
        train_path, idx_train=train_idx, idx_val=val_idx
    )
    test = scaler.transform(load(test_path))
    if additional:
        test_path = data_path / f"embeddings/test_{additional.lower()}.npy"
        train_path = data_path / f"embeddings/train_{additional.lower()}.npy"
        train_a, val_a, scaler_a = load_embeddings_npy(
            train_path, idx_train=train_idx, idx_val=val_idx
        )
        test_a = scaler_a.transform(load(test_path))
        train = fusion_concat(train, train_a)
        val = fusion_concat(val, val_a)
        test = fusion_concat(test, test_a)
    return train, val, test  # type: ignore


@timeit("MLP", results)
def train_keras(
    X_train: ndarray,
    y_train: ndarray,
    X_val: ndarray,
    y_val: ndarray,
    X_test: ndarray,
    name: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\nTuning and fitting: MLP")
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
    best_model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop])
    y_pred = best_model.predict(X_val)
    y_pred_classes = y_pred.argmax(axis=1)
    y_test = best_model.predict(X_test)
    y_test_classes = y_test.argmax(axis=1)
    report = classification_report(
        y_val, y_pred_classes, digits=4, output_dict=True
    )
    assert isinstance(report, dict)
    results["MLP"] = report
    results["MLP"]["Hyperparameters"] = tuner.get_best_hyperparameters()[
        0
    ].values
    print(
        "#### Report for MLP:\n####",
        results,
    )
    return results, {"MLP": y_test_classes}


def train_classificators(
    X_train: ndarray,
    y_train: ndarray,
    X_val: ndarray,
    y_val: ndarray,
    X_test: ndarray,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    classifiers = get_classifiers(random_state)
    for name, model in classifiers.items():
        timed_training = timeit(name, results)(train_classificator)
        timed_training(X_train, y_train, X_val, y_val, X_test, name, model)
    return results, predictions


def train_classificator(X_train, y_train, X_val, y_val, X_test, name, model):
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
        args.embedding + "+" + args.additional
        if args.additional
        else args.embedding
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
        X_train, y_train, X_val, y_val, X_test, args
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
