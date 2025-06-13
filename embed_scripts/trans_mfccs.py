import os
import json
import numpy as np
import pandas as pd
import librosa
import time
from pathlib import Path
from tqdm import tqdm

# ================== CONFIGURACION ====================

# Paths de los datos
DATA_DIR = Path("./data/public_data")
SEGMENTS_TRAIN_DIR = DATA_DIR / "segments_train"
SEGMENTS_TEST_DIR = DATA_DIR / "segments_test"
CSV_TRAIN = DATA_DIR / "SatiSPeech_phase_2_train_public.csv"
CSV_TEST = DATA_DIR / "SatiSPeech_phase_2_test_public.csv"

# Paths de salida
OUTPUT_DIR = DATA_DIR / "mfcc_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_MFCC_STATS = OUTPUT_DIR / "train_mfcc_stats.npy"
TRAIN_MFCC_PROSODIC = OUTPUT_DIR / "train_mfcc_prosodic.npy"
TRAIN_MFCC_FULL = OUTPUT_DIR / "train_mfcc_full.npy"

TEST_MFCC_STATS = OUTPUT_DIR / "test_mfcc_stats.npy"
TEST_MFCC_PROSODIC = OUTPUT_DIR / "test_mfcc_prosodic.npy"
TEST_MFCC_FULL = OUTPUT_DIR / "test_mfcc_full.npy"

ERRORS_TRAIN = OUTPUT_DIR / "errores_train.json"
ERRORS_TEST = OUTPUT_DIR / "errores_test.json"

# Sampling rate objetivo
TARGET_SAMPLING_RATE = 16000

# MFCC parametros
N_MFCC = 13

# Cada cuántos audios imprimir estado
PRINT_EVERY = 100

# =======================================================

# Funciones auxiliares


def load_audio(path, target_sr=16000):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y


def extract_mfcc_stats(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return features


def extract_prosodic_features(y, sr=16000):
    zcr = librosa.feature.zero_crossing_rate(y)
    energy = librosa.feature.rms(y=y)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
    features = np.array(
        [zcr.mean(), zcr.std(), energy.mean(), energy.std(), pitch_mean]
    )
    return features


def extract_mfcc_prosodic(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features_mfcc = np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])
    prosodic = extract_prosodic_features(y, sr)
    features = np.hstack([features_mfcc, prosodic])
    return features


def extract_full_features(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features_mfcc = np.hstack(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta_mfcc.mean(axis=1),
            delta_mfcc.std(axis=1),
            delta2_mfcc.mean(axis=1),
            delta2_mfcc.std(axis=1),
        ]
    )
    prosodic = extract_prosodic_features(y, sr)
    features = np.hstack([features_mfcc, prosodic])
    return features


def process_split(
    csv_path: os.PathLike,
    audio_dir: os.PathLike,
    id_column: str,
    output_paths: list[os.PathLike],
    errors_path: os.PathLike,
    split_name: str = "Split",
):
    df = pd.read_csv(csv_path)
    mfcc_stats_embeddings = []
    mfcc_prosodic_embeddings = []
    full_embeddings = []
    errors = {}

    processed_count = 0
    error_count = 0

    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_filename = row[id_column]
        audio_path = os.path.join(audio_dir, audio_filename)

        try:
            y = load_audio(audio_path)
            mfcc_stats = extract_mfcc_stats(
                y, sr=TARGET_SAMPLING_RATE, n_mfcc=N_MFCC
            )
            mfcc_prosodic = extract_mfcc_prosodic(
                y, sr=TARGET_SAMPLING_RATE, n_mfcc=N_MFCC
            )
            full = extract_full_features(
                y, sr=TARGET_SAMPLING_RATE, n_mfcc=N_MFCC
            )

            mfcc_stats_embeddings.append(mfcc_stats)
            mfcc_prosodic_embeddings.append(mfcc_prosodic)
            full_embeddings.append(full)

        except Exception as e:
            errors[audio_filename] = str(e)
            error_count += 1
            print(f"[{split_name}] Error en {audio_filename}: {e}")

        processed_count += 1

        if processed_count % PRINT_EVERY == 0:
            print(
                f"[{split_name}] Procesados: {processed_count} audios | Fallos: {error_count}"
            )

    np.save(output_paths[0], np.array(mfcc_stats_embeddings))
    np.save(output_paths[1], np.array(mfcc_prosodic_embeddings))
    np.save(output_paths[2], np.array(full_embeddings))

    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{split_name}] Tiempo total: {elapsed_time / 60:.2f} minutos")


# =============== PROCESAMIENTO PRINCIPAL ================

if __name__ == "__main__":
    print("Procesando TRAIN...")
    process_split(
        CSV_TRAIN,
        SEGMENTS_TRAIN_DIR,
        "id",
        [TRAIN_MFCC_STATS, TRAIN_MFCC_PROSODIC, TRAIN_MFCC_FULL],
        ERRORS_TRAIN,
        split_name="Train",
    )

    print("Procesando TEST...")
    process_split(
        CSV_TEST,
        SEGMENTS_TEST_DIR,
        "uid",
        [TEST_MFCC_STATS, TEST_MFCC_PROSODIC, TEST_MFCC_FULL],
        ERRORS_TEST,
        split_name="Test",
    )

    print("\n✅ Proceso completado.")
