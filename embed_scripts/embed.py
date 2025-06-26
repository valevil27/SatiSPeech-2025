from argparse import ArgumentParser
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

TRAIN_NAME = "SatiSPeech_phase_2_train_public.csv"
TEST_NAME = "SatiSPeech_phase_2_test_public.csv"
SAMPLING_RATE = 16_000


class Embedding(StrEnum):
    FASTTEXT = auto()
    WORD2VEC = auto()
    ROBERTA = auto()
    MFCC = auto()
    HUBERT = auto()
    W2V2 = auto()
    W2V2BERT = auto()


@dataclass
class Args:
    data_dir: Path
    logs_dir: Path
    output_dir: Path
    embedding: Embedding
    print_frequency: int

    def __post_init__(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--embedding",
        type=Embedding,
        required=True,
    )
    parser.add_argument("-d", "--data-dir", type=Path, default="data/public_data")
    parser.add_argument("-l", "--logs-dir", type=Path, default="logs")
    parser.add_argument("-o", "--output-dir", type=Path, default="data/embeddings")
    parser.add_argument("-f", "--print-state-frequency", type=int, default=100)
    args = parser.parse_args()
    return Args(
        embedding=args.embedding,
        data_dir=args.data_dir,
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        print_frequency=args.print_state_frequency,
    )


def main():
    a = parse_args()
    train_path = a.data_dir / TRAIN_NAME
    test_path = a.data_dir / TEST_NAME
    out_train_path = a.output_dir / f"train_{a.embedding.value}.npy"
    out_test_path = a.output_dir / f"test_{a.embedding.value}.npy"

    match a.embedding:
        case Embedding.FASTTEXT:
            import trans_fasttext

            trans_fasttext.process_embeddings(
                csv_path=train_path,
                output_path=out_train_path,
                split_name="Train",
            )
            trans_fasttext.process_embeddings(
                csv_path=test_path,
                output_path=out_test_path,
                split_name="Test",
            )
        case Embedding.WORD2VEC:
            import trans_word2vec

            trans_word2vec.process_embeddings(
                csv_path=train_path,
                output_path=out_train_path,
                split_name="Train",
            )
            trans_word2vec.process_embeddings(
                csv_path=test_path,
                output_path=out_test_path,
                split_name="Test",
            )
        case Embedding.ROBERTA:
            import trans_roberta

            trans_roberta.process_embeddings(
                csv_path=train_path,
                output_path=out_train_path,
                split_name="Train",
            )
            trans_roberta.process_embeddings(
                csv_path=test_path,
                output_path=out_test_path,
                split_name="Test",
            )
        case Embedding.MFCC:
            train_paths = [
                a.output_dir / f"train_mfcc-{k}.npy"
                for k in ["stats", "prosodic", "full"]
            ]
            test_paths = [
                a.output_dir / f"test_mfcc-{k}.npy"
                for k in ["stats", "prosodic", "full"]
            ]
            import trans_mfccs

            trans_mfccs.process_embeddings(
                csv_path=train_path,
                audio_dir=a.data_dir / "segments_train",
                id_column="id",
                output_paths=train_paths,
                errors_path=a.logs_dir / "train_mfcc.json",
                split_name="Train",
            )
            trans_mfccs.process_embeddings(
                csv_path=test_path,
                audio_dir=a.data_dir / "segments_test",
                id_column="uid",
                output_paths=test_paths,
                errors_path=a.logs_dir / "test_mfcc.json",
                split_name="Test",
            )
        case Embedding.HUBERT:
            import trans_hubert

            trans_hubert.process_embeddings(
                csv_path=train_path,
                audio_dir=a.data_dir / "segments_train",
                id_column="id",
                output_cls=a.output_dir / "train_hubert-cls.npy",
                output_mean=a.output_dir / "train_hubert-mean.npy",
                errors_path=a.logs_dir / "train_hubert.json",
                split_name="Train",
            )
            trans_hubert.process_embeddings(
                csv_path=test_path,
                audio_dir=a.data_dir / "segments_test",
                id_column="uid",
                output_cls=a.output_dir / "test_hubert-cls.npy",
                output_mean=a.output_dir / "test_hubert-mean.npy",
                errors_path=a.logs_dir / "test_hubert.json",
                split_name="Test",
            )
        case Embedding.W2V2:
            import trans_wav2vec2

            trans_wav2vec2.process_embeddings(
                csv_path=train_path,
                audio_dir=a.data_dir / "segments_train",
                id_column="id",
                output_cls=a.output_dir / "train_wav2vec2-cls.npy",
                output_mean=a.output_dir / "train_wav2vec2-mean.npy",
                errors_path=a.logs_dir / "train_wav2vec2.json",
                split_name="Train",
            )
            trans_wav2vec2.process_embeddings(
                csv_path=test_path,
                audio_dir=a.data_dir / "segments_test",
                id_column="uid",
                output_cls=a.output_dir / "test_wav2vec2-cls.npy",
                output_mean=a.output_dir / "test_wav2vec2-mean.npy",
                errors_path=a.logs_dir / "test_wav2vec2.json",
                split_name="Test",
            )
        case Embedding.W2V2BERT:
            import trans_wav2vec2BERT

            trans_wav2vec2BERT.process_embeddings(
                csv_path=train_path,
                audio_dir=a.data_dir / "segments_train",
                id_column="id",
                output_cls=a.output_dir / "train_wav2vec2bert-cls.npy",
                output_mean=a.output_dir / "train_wav2vec2bert-mean.npy",
                errors_path=a.logs_dir / "train_wav2vec2bert.json",
                split_name="Train",
            )
            trans_wav2vec2BERT.process_embeddings(
                csv_path=test_path,
                audio_dir=a.data_dir / "segments_test",
                id_column="uid",
                output_cls=a.output_dir / "test_wav2vec2bert-cls.npy",
                output_mean=a.output_dir / "test_wav2vec2bert-mean.npy",
                errors_path=a.logs_dir / "test_wav2vec2bert.json",
                split_name="Test",
            )


if __name__ == "__main__":
    main()
