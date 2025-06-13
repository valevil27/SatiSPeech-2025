from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from enum import StrEnum, auto

import trans_mfccs

TRAIN_NAME = "SatiSPeech_phase_2_train_public.csv"
TEST_NAME = "SatiSPeech_phase_2_test_public.csv"
SAMPLING_RATE = 16_000


class Embedding(StrEnum):
    MFCC = auto()
    FASTTEXT = auto()
    WORD2VEC = auto()
    ROBERTA = auto()
    HUBERT_CLS = auto()
    HUBERT_MEAN = auto()
    W2V2_CLS = auto()
    W2V2_MEAN = auto()


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
    parser.add_argument(
        "-d", "--data-dir", type=Path, default="data/public_data"
    )
    parser.add_argument("-l", "--logs-dir", type=Path, default="logs")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default="data/embeddings"
    )
    parser.add_argument("-f", "--print-state-frequency", type=int, default=100)
    args = parser.parse_args()
    return Args(
        embedding=args.embedding,
        data_dir=args.data_dir,
        logs_dir=args.logs_dir,
        output_dir=args.output_dir,
        print_frequency=args.frequency,
    )


def main():
    a = parse_args()
    match a.embedding:
        case Embedding.MFCC:
            trans_mfccs.process_split(
                csv_path=a.data_dir / TRAIN_NAME,
                audio_dir= a.data_dir / "segments_train",
                id_column="id",
                output_paths=[
                    a.output_dir / "train_mfcc_stats.npy",
                    a.output_dir / "train_mfcc_prosodic.npy",
                    a.output_dir / "train_mfcc_full.npy",
                ],
                errors_path=a.logs_dir / "train_mfcc.json",
            )
            trans_mfccs.process_split(
                csv_path=a.data_dir / TEST_NAME,
                audio_dir= a.data_dir / "segments_test",
                id_column="uid",
                output_paths=[
                    a.output_dir / "test_mfcc_stats.npy",
                    a.output_dir / "test_mfcc_prosodic.npy",
                    a.output_dir / "test_mfcc_full.npy",
                ],
                errors_path=a.logs_dir / "test_mfcc.json",
            )



if __name__ == "__main__":
    main()
