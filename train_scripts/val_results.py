import argparse
from dataclasses import dataclass, field
from enum import StrEnum
import numpy as np
import json
from pathlib import Path
import random
from keras_tuner import HyperParameters
from numpy import ndarray
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from single_script import (
    Embedding,
    get_labels,
    load_dfs,
    load_embeddings,
)
from combi_script import Method, fuse_embeddings
from keras_utils import build_model, get_early_stop


IDX_PATH = Path(__file__).parent / "idx.txt"


def load_idx(path: Path) -> tuple[ndarray, ndarray]:
    if not path.exists():
        print(f"Creating index file at {path}")
        val_idx = random.sample(range(6000), 500)
        IDX_PATH.touch(exist_ok=False)
        IDX_PATH.write_text("\n".join(map(str, val_idx)))
    print(f"Loading index file at {path}")
    with open(path, "r") as f:
        val_idx = set(map(int, f.read().splitlines()))
    train_idx = set(range(6000)) - val_idx
    return np.array(list(train_idx)), np.array(list(val_idx))


class Model(StrEnum):
    DNN = "DNN"
    LogisticRegression = "LogisticRegression"
    SVM = "SVM"
    RandomForest = "RandomForest"


@dataclass
class Args:
    model: Model
    input_json: Path
    data_dir: Path
    output_dir: Path
    text_embeddings: list[Embedding] = field(default_factory=list)
    audio_embeddings: list[Embedding] = field(default_factory=list)
    method: Method = Method.CONCAT

    def __post_init__(self):
        assert self.data_dir.exists(), (
            f"data directory {self.data_dir} does not exist"
        )
        self.text_embeddings, self.audio_embeddings, self.method = (
            Args.parse_name(self.input_json.stem)
        )
        if not self.input_json.exists():
            raise ValueError(
                f"The results path {self.input_json} does not exist."
            )
        if self.input_json.suffix != ".json":
            raise ValueError(
                f"The results path {self.input_json} must be a JSON file."
            )
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        if self.model == Model.DNN:
            self.hyperparameters = self.load_hyperparameters_kt()
        else:
            self.hyperparameters = self.load_hyperparameters_sk()
        self.name = self.model.value
        self.name += f"+{'_'.join(map(str, self.text_embeddings))}"
        self.name += f"+{'_'.join(map(str, self.audio_embeddings))}"
        self.name += f"+{self.method.value}"

    @staticmethod
    def parse_name(
        name: str,
    ) -> tuple[list[Embedding], list[Embedding], Method]:
        te = list()
        ae = list()
        text_str, audio_str, method_str = name.split("+")
        te = [Embedding(m) for m in text_str.split("_")]
        ae = [Embedding(m) for m in audio_str.split("_")]
        return te, ae, Method(method_str)

    def load_hyperparameters_sk(self) -> dict:
        with open(self.input_json, "r") as f:
            return json.load(f)

    def load_hyperparameters_kt(self) -> HyperParameters:
        hyperparameters = HyperParameters()
        with open(self.input_json, "r") as f:
            for hp, val in json.load(f)[self.model.value][
                "Hyperparameters"
            ].items():
                hyperparameters.values[hp] = val
        return hyperparameters


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Script that trains a model a using the provided result file hyperparameters and returns the index of the missclassified samples."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("data/public_data"),
        help="Path to the data directory. Default: ./data/public_data",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Model,
        required=True,
        choices=Model,
        help="Model to use for classification.",
    )
    parser.add_argument(
        "--input-json",
        "-i",
        type=Path,
        required=True,
        help="Path to the JSON file with the results of the experiment with the model hyperparameters.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=False,
        default=Path("results/misses"),
        help="Output directory for the results JSON files (created if it does not exist). Default: ./results/misses.",
    )
    args = parser.parse_args()
    return Args(
        data_dir=args.data_dir,
        model=args.model,
        input_json=args.input_json,
        output_dir=args.output_dir,
    )


def main():
    args = parse_args()
    print(args)
    print(f"""\nStrategy to follow:
\t- Text embedding: {", ".join(map(str, args.text_embeddings))}
\t- Audio embedding: {", ".join(map(str, args.audio_embeddings))}
\t- Fusion method: {args.method.value}
\t- Classifier: {args.model.value}
\t- Hyperparameters:""")
    for k, v in args.hyperparameters.values.items():
        print(f"\t\t> {k}: {v}")
    train_df, test_df = load_dfs(args.data_dir)
    train_idx, val_idx = load_idx(IDX_PATH)
    X_train_text, X_val_text, X_test_text = load_embeddings(
        args.data_dir,
        train_idx,
        val_idx,
        args.text_embeddings[0],
        None if len(args.text_embeddings) == 1 else args.text_embeddings[1],
    )
    X_train_audio, X_val_audio, X_test_audio = load_embeddings(
        args.data_dir,
        train_idx,
        val_idx,
        args.audio_embeddings[0],
        None if len(args.audio_embeddings) == 1 else args.audio_embeddings[1],
    )
    y_train, y_val = get_labels(train_df, train_idx, val_idx)
    X_train, X_val, X_test = fuse_embeddings(
        X_train_text,
        X_train_audio,
        X_val_text,
        X_val_audio,
        X_test_text,
        X_test_audio,
        y_train,
        y_val,
        args.method,
    )
    match args.model:
        case Model.DNN:
            assert isinstance(args.hyperparameters, HyperParameters)
            model = build_model(X_train, y_train)(args.hyperparameters)
            model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[get_early_stop()])
            prediction = model.predict(X_val).argmax(axis=1)
        case Model.LogisticRegression:
            assert isinstance(args.hyperparameters, dict)
            model = LogisticRegression(**args.hyperparameters)
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
        case Model.SVM:
            assert isinstance(args.hyperparameters, dict)
            model = SVC(**args.hyperparameters)
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
        case Model.LogisticRegression:
            assert isinstance(args.hyperparameters, dict)
            model = RandomForestClassifier(**args.hyperparameters)
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)
        case _:
            raise ValueError(f"Model {args.model} not valid")
    df = pd.DataFrame.from_dict(
        {
            "true": y_val,
            "prediction": prediction,
        },
        orient="index",
    ).transpose()
    df["id"] = train_df["id"].str.removesuffix(".mp3")
    ids_with_errors = df[df["true"] != df["prediction"]]["id"].values.tolist()
    with open(args.output_dir / f"{args.name}.json", "w") as f:
        json.dump({
            "text_embeddings": args.text_embeddings,
            "audio_embeddings": args.audio_embeddings,
            "method": args.method.value,
            "model": args.model.value,
            "hyperparameters": args.hyperparameters.values,
            "ids_with_errors": ids_with_errors,
        }, f)
    print("Results saved to", args.output_dir / f"{args.name}.json")


if __name__ == "__main__":
    main()
