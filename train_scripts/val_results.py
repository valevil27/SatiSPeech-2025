import argparse
from dataclasses import dataclass, field
from enum import StrEnum
import numpy as np
import json
from pathlib import Path
import random
from keras_tuner import HyperParameters
from numpy import ndarray
from single_script import Embedding, load_dfs
from combi_script import Method


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
        if self.model == Model.DNN:
            self.hyperparameters = self.load_hyperparameters_kt()
        else:
            self.hyperparameters = self.load_hyperparameters_sk()

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


if __name__ == "__main__":
    main()
