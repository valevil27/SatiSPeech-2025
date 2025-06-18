from dataclasses import dataclass
from enum import StrEnum
import numpy as np
import json
from pathlib import Path
import random
from keras_tuner import HyperParameters
from numpy import ndarray
from single_script import Embedding
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

def parse_name(name: str) -> tuple[list[Embedding], list[Embedding], Method]:
    te = list()
    ae = list()
    text_str, audio_str, method_str = name.split("+")
    te = [Embedding(m) for m in text_str.split("_")]
    ae = [Embedding(m) for m in audio_str.split("_")]
    return te, ae, Method(method_str)


class Model(StrEnum):
    DNN = "DNN"
    LogisticRegression = "LogisticRegression"
    SVM = "SVM"
    RandomForest = "RandomForest"


@dataclass
class Args:
    model: Model
    results_path: Path

    def __post_init__(self):
        self.text_embeddings, self.audio_embeddings, self.method = parse_name(
            self.results_path.stem
        )
        if not self.results_path.exists():
            raise ValueError(
                f"The results path {self.results_path} does not exist."
            )
        if self.results_path.suffix != ".json":
            raise ValueError(
                f"The results path {self.results_path} must be a JSON file."
            )
        self.hyperparameters = HyperParameters()
        with open(self.results_path, "r") as f:
            for hp, val in json.load(f)[self.model.value]["Hyperparameters"].items():
                self.hyperparameters.values[hp] = val

def main():
    results_path = (
        Path.cwd() / "results/combi/fasttext+hubert_cls_attention.json"
    )
    args = Args(model=Model.DNN, results_path=results_path)
    print(args)
    print(args.hyperparameters.values)
    train_idx, val_idx = load_idx(IDX_PATH)
    print(val_idx)


if __name__ == "__main__":
    main()
