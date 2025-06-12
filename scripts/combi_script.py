from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from numpy import ndarray
from fusion_utils import (
    fusion_attention,
    fusion_concat,
    fusion_mean,
    fusion_weighted,
    search_best_weighted_fusion,
)
from text_script import (
    get_labels,
    get_splits_idx,
    load_dfs,
    load_embeddings,
    save_results,
    train_classificators,
    train_keras,
)

results = {}
predictions = {}
valid_embeddings = {
    "text": [
        "fasttext",
        "mfcc_full",
        "mfcc_prosodic",
        "mfcc_stats",
        "word2vec",
    ],
    "audio": ["hubert_cls", "hubert_mean", "w2v2_cls", "w2v2_mean"],
}
valid_methods = ["concat", "mean", "weighted", "attention"]


@dataclass
class Args:
    name: str
    text_embedding: str
    text_additional: Optional[str]
    audio_embedding: str
    audio_additional: Optional[str]
    method: str
    train_size: int
    val_size: int
    random_state: int
    output_path: Path

    def __post_init__(self):
        self.text_embedding = self.text_embedding.lower()
        assert Args.get_embedding_type(self.text_embedding) == "text", (
            "text embedding must be a text embedding"
        )
        if self.text_additional:
            self.text_additional = self.text_additional.lower()
            assert Args.get_embedding_type(self.text_additional) == "text", (
                "text embedding must be a text embedding"
            )
        self.audio_embedding = self.audio_embedding.lower()
        assert Args.get_embedding_type(self.audio_embedding) == "audio", (
            "audio embedding must be an audio embedding"
        )
        if self.audio_additional:
            self.audio_additional = self.audio_additional.lower()
            assert Args.get_embedding_type(self.audio_additional) == "audio", (
                "audio embedding must be an audio embedding"
            )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_path / f"combi_{self.name}"

    @staticmethod
    def get_embedding_type(embedding: str) -> str:
        for k, v in valid_embeddings.items():
            if embedding in v:
                return k
        raise ValueError("embedding not supported")


def fuse_embeddings(
    X_train_text: ndarray,
    X_train_audio: ndarray,
    X_val_text: ndarray,
    X_val_audio: ndarray,
    X_test_text: ndarray,
    X_test_audio: ndarray,
    y_train: ndarray,
    y_val: ndarray,
    method: str,
) -> tuple[ndarray, ndarray, ndarray]:
    f: Callable
    match method:
        case "concat":
            f = fusion_concat
        case "mean":
            f = fusion_mean
        case "weighted":
            [w_a, w_b], _ = search_best_weighted_fusion(
                X_train_text,
                X_train_audio,
                X_val_text,
                X_val_audio,
                y_train,
                y_val,
            )
            f = partial(fusion_weighted, weight_a=w_a, weight_b=w_b)
        case "attention":
            f = fusion_attention
        case _:
            raise ValueError("fusion method not valid")
    X_train = f(X_train_text, X_train_audio)
    X_val = f(X_val_text, X_val_audio)
    X_test = f(X_test_text, X_test_audio)
    return X_train, X_val, X_test


def parse_args() -> Args:
    parser = ArgumentParser(
        description="Script that trains several models using the provided embeddings data as input and produces several results."
    )
    parser.add_argument(
        "--name", "-n", type=str, required=True, help="Experiment name."
    )
    parser.add_argument(
        "--text-embedding",
        "-te",
        type=str,
        required=True,
        help='Name of the embedding to use. The embedding must be previously created in the "embeddings" folder, in ".npy" format for both the test and train sets, along with the data CSV files. The name format should be "test_<embedding>.npy."',
    )
    parser.add_argument(
        "--text-additional",
        "-ta",
        type=str,
        required=False,
        help="Name of the additional embedding to use along with the main one. The format is the same as for the main embedding. If not present, only the main embedding is used.",
    )
    parser.add_argument(
        "--audio-embedding",
        "-ae",
        type=str,
        required=True,
        help='Name of the embedding to use. The embedding must be previously created in the "embeddings" folder, in ".npy" format for both the test and train sets, along with the data CSV files. The name format should be "test_<embedding>.npy."',
    )
    parser.add_argument(
        "--audio-additional",
        "-aa",
        type=str,
        required=False,
        help="Name of the additional embedding to use along with the main one. The format is the same as for the main embedding. If not present, only the main embedding is used.",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=valid_methods,
        type=str,
        default="concat",
        help="Method used to combine text and audio embeddings. Allowed methods are concat, mean, weighted and attention. By default, uses concat",
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
        text_embedding=args.text_embedding,
        text_additional=args.text_additional,
        audio_embedding=args.audio_embedding,
        audio_additional=args.audio_additional,
        method=args.method,
        output_path=args.output,
        random_state=args.random_state,
        train_size=args.train_size,
        val_size=args.val_size,
    )


def main():
    args = parse_args()
    data_path = Path.cwd() / "data/public_data"
    text_embeddings = (
        args.text_embedding + "+" + args.text_additional
        if args.text_additional
        else args.text_embedding
    )
    audio_embeddings = (
        args.audio_embedding + "+" + args.audio_additional
        if args.audio_additional
        else args.audio_embedding
    )
    print(
        f"Experimento {args.name}.\nUsando embeddings {text_embeddings} con {audio_embeddings} mediante {args.method}."
    )
    print(f"Random state = {args.random_state}")
    print(f"Directorio de salida: {args.output_path}")
    train_df, test_df = load_dfs(data_path)
    train_idx, val_idx = get_splits_idx(
        train_df, args.train_size, args.val_size, args.random_state
    )
    X_train_text, X_val_text, X_test_text = load_embeddings(
        data_path, train_idx, val_idx, args.text_embedding, args.text_additional
    )
    X_train_audio, X_val_audio, X_test_audio = load_embeddings(
        data_path,
        train_idx,
        val_idx,
        args.audio_embedding,
        args.audio_additional,
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
