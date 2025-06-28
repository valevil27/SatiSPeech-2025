from __future__ import annotations
import keras
import numpy as np
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from typing import Callable, Any
from sklearn.metrics import classification_report
from numpy import ndarray
from keras_utils import (
    build_attention_model,
    get_early_stop,
    get_tuner,
    get_attention_embeddings,
)
from fusion_utils import (
    fusion_attention,
    fusion_concat,
    fusion_mean,
    fusion_weighted,
    search_best_weighted_fusion,
)
from single_script import (
    Model,
    get_labels,
    get_splits_idx,
    load_dfs,
    load_embeddings,
    save_results,
    train_classificators,
    train_keras,
    Embedding,
)
from classif_utils import timeit

results = {}
predictions = {}


class Method(StrEnum):
    CONCAT = auto()
    MEAN = auto()
    WEIGHTED = auto()
    ATTENTION = auto()
    ALL = auto()

    @staticmethod
    def valid_methods() -> list[Method]:
        return [m for m in Method if m != Method.ALL]


@dataclass
class Args:
    text_embeddings: list[Embedding]
    audio_embeddings: list[Embedding]
    data_dir: Path
    methods: list[Method]
    train_size: int
    val_size: int
    random_state: int
    output_path: Path
    models: list[Model]

    def __post_init__(self):
        """
        Lowercases embedding names, creates a name for the project and creates output directory if needed.
        """
        assert self.data_dir.exists(), f"data directory {self.data_dir} does not exist"
        if Method.ALL in self.methods:
            self.methods = Method.valid_methods()
        if Model.ALL in self.models:
            self.models = Model.valid_models()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.name = f"{'_'.join(t.value for t in self.text_embeddings)}+"
        self.name += f"{'_'.join(a.value for a in self.audio_embeddings)}"
        self.output_path = self.output_path / f"{self.name}"


def fuse_embeddings(
    X_train_audio: ndarray,
    X_train_text: ndarray,
    X_val_audio: ndarray,
    X_val_text: ndarray,
    X_test_audio: ndarray,
    X_test_text: ndarray,
    y_train: ndarray,
    y_val: ndarray,
    method: Method,
) -> tuple[ndarray, ndarray, ndarray]:
    f: Callable
    match method:
        case Method.CONCAT:
            f = fusion_concat
        case Method.MEAN:
            f = fusion_mean
        case Method.WEIGHTED:
            [w_a, w_b], _ = search_best_weighted_fusion(
                X_train_audio,
                X_train_text,
                X_val_audio,
                X_val_text,
                y_train,
                y_val,
            )
            f = partial(fusion_weighted, weight_a=w_a, weight_b=w_b)
        case Method.ATTENTION:
            f = fusion_attention
        case _:
            raise ValueError("fusion method not valid")
    X_train = f(X_train_audio, X_train_text)
    X_val = f(X_val_audio, X_val_text)
    X_test = f(X_test_audio, X_test_text)
    return X_train, X_val, X_test


@timeit("ATTENTION_DNN", results)
def train_attention(
    X_train_text: np.ndarray,
    X_train_audio: np.ndarray,
    X_val_text: np.ndarray,
    X_val_audio: np.ndarray,
    X_test_text: np.ndarray,
    X_test_audio: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    name: str,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any], keras.Model]:
    """Tunes and trains an attention-based model"""
    print("\nTuning attention-based model...")
    builder = build_attention_model(
        X_train_text.shape[1],
        X_train_audio.shape[1],
        y_train.shape[0],
    )
    tuner = get_tuner(builder, name, random_state)
    tuner.search(
        [X_train_text, X_train_audio],
        y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[get_early_stop()],
        verbose=1,
    )
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = builder(best_hps)
    best_model.fit(
        [X_train_text, X_train_audio],
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[get_early_stop()],
        verbose=0,  # type: ignore
    )
    y_pred = best_model.predict([X_val_text, X_val_audio])
    y_pred_classes = y_pred.argmax(axis=1)
    y_test = best_model.predict([X_test_text, X_test_audio])
    y_test_classes = y_test.argmax(axis=1)
    report = classification_report(y_val, y_pred_classes, output_dict=True)
    assert isinstance(report, dict)
    results["ATTENTION_DNN"] = report
    results["ATTENTION_DNN"]["Hyperparameters"] = best_hps.values
    predictions["ATTENTION_DNN"] = y_test_classes
    print("#### Report for Attention DNN:\n", report)
    return results, predictions, best_model


def train_attention_class(
    X_train_text: np.ndarray,
    X_train_audio: np.ndarray,
    X_val_text: np.ndarray,
    X_val_audio: np.ndarray,
    X_test_text: np.ndarray,
    X_test_audio: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    models: list[Model],
    random_state: int,
    att_model: keras.Sequential,
) -> tuple[dict[str, Any], dict[str, Any]]:
    emb_train = get_attention_embeddings(att_model, X_train_text, X_train_audio)
    emb_val = get_attention_embeddings(att_model, X_val_text, X_val_audio)
    emb_test = get_attention_embeddings(att_model, X_test_text, X_test_audio)
    class_results, class_predictions = train_classificators(
        emb_train,
        y_train,
        emb_val,
        y_val,
        emb_test,
        random_state,
        models,
    )
    return class_results, class_predictions


def parse_args() -> Args:
    parser = ArgumentParser(
        description="Script that trains several models using the provided embeddings data as input and produces several results."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("data/public_data"),
        help="Path to the data directory. Default: ./data/public_data",
    )
    parser.add_argument(
        "--text-embeddings",
        "-te",
        type=Embedding,
        nargs="+",
        choices=[e for e in Embedding if e.type() == "text"],
        required=True,
        help='Name of the embedding to use. The embedding must be previously created in the "embeddings" folder, in ".npy" format for both the test and train sets, along with the data CSV files. The name format should be "test_<embedding>.npy."',
    )
    parser.add_argument(
        "--audio-embeddings",
        "-ae",
        type=Embedding,
        nargs="+",
        choices=[e for e in Embedding if e.type() == "audio"],
        required=True,
        help='Name of the embedding to use. The embedding must be previously created in the "embeddings" folder, in ".npy" format for both the test and train sets, along with the data CSV files. The name format should be "test_<embedding>.npy."',
    )
    parser.add_argument(
        "--methods",
        "-m",
        type=Method,
        nargs="+",
        choices=Method,
        default=["concat"],
        help="Method used to fuse text and audio embeddings. Allowed one or more methods from concat, mean, weighted, attention and all. By default, uses concat",
    )
    parser.add_argument(
        "--classifiers",
        "-c",
        nargs="+",
        choices=Model,
        default=[Model.ALL],
        help="Classifier models to use for classification. Can select several models separated by spaces. By default, uses all models.",
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
        default=Path("results/combi"),
        help="Output directory for the results JSON files (created if it does not exist). Default: ./results.",
    )
    args = parser.parse_args()
    return Args(
        text_embeddings=args.text_embeddings,
        audio_embeddings=args.audio_embeddings,
        data_dir=args.data_dir,
        methods=args.methods,
        models=args.classifiers,
        output_path=args.output,
        random_state=args.random_state,
        train_size=args.train_size,
        val_size=args.val_size,
    )


def main():
    args = parse_args()
    text_embeddings = "+".join(t.value for t in args.text_embeddings)
    audio_embeddings = "+".join(a.value for a in args.audio_embeddings)
    print(f"Random state = {args.random_state}")
    print(f"Directorio de salida: {args.output_path}")
    train_df, test_df = load_dfs(args.data_dir)
    train_idx, val_idx = get_splits_idx(
        train_df, args.train_size, args.val_size, args.random_state
    )
    X_train_text, X_val_text, X_test_text = load_embeddings(
        args.data_dir,
        train_idx,
        val_idx,
        args.text_embeddings[0],
        args.text_embeddings[1] if len(args.text_embeddings) > 1 else None,
    )
    X_train_audio, X_val_audio, X_test_audio = load_embeddings(
        args.data_dir,
        train_idx,
        val_idx,
        args.audio_embeddings[0],
        args.audio_embeddings[1] if len(args.audio_embeddings) > 1 else None,
    )
    y_train, y_val = get_labels(train_df, train_idx, val_idx)
    for method in args.methods:
        print(f"Training {text_embeddings} with {audio_embeddings} using {method}.")
        if method == Method.ATTENTION:
            assert Model.DNN in args.models, (
                "DNN model is required for attention-based training"
            )
            keras_results, keras_preds, model = train_attention(
                X_train_text,
                X_train_audio,
                X_val_text,
                X_val_audio,
                X_test_text,
                X_test_audio,
                y_train,
                y_val,
                f"{args.name}_{method.value}",
                args.random_state,
            )
            class_results, class_preds = train_attention_class(
                X_train_text,
                X_train_audio,
                X_val_text,
                X_val_audio,
                X_test_text,
                X_test_audio,
                y_train,
                y_val,
                args.models,
                args.random_state,
                model,
            )
        else:
            X_train, X_val, X_test = fuse_embeddings(
                X_train_audio,
                X_train_text,
                X_val_audio,
                X_val_text,
                X_test_audio,
                X_test_text,
                y_train,
                y_val,
                method,
            )
            keras_results, keras_preds = dict(), dict()
            if Model.DNN in args.models:
                keras_results, keras_preds = train_keras(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    f"{args.name}_{method.value}",
                    args.random_state,
                )
            class_results, class_preds = train_classificators(
                X_train, y_train, X_val, y_val, X_test, args.random_state, args.models
            )
        save_results(
            test_df,
            keras_results | class_results,
            keras_preds | class_preds,
            args.output_path.with_name(args.output_path.name + "+" + method.value),
        )


if __name__ == "__main__":
    main()
