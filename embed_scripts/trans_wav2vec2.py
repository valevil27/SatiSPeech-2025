import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import time
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model  # type: ignore
from trans_hubert import load_audio, extract_embeddings

# ================== CONFIGURACION ====================

# Modelo
W2V2_MODEL_NAME = "facebook/wav2vec2-base-960h"

# Dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sampling rate objetivo
TARGET_SAMPLING_RATE = 16000

# Cada cuántos audios imprimir estado
PRINT_EVERY = 100

# =======================================================

# Set backend para MP3
try:
    torchaudio.set_audio_backend("sox_io")
except Exception as _:
    torchaudio.set_audio_backend("ffmpeg")

# Cargar modelos y extractores
w2v2_processor = Wav2Vec2Processor.from_pretrained(W2V2_MODEL_NAME)
w2v2_model = Wav2Vec2Model.from_pretrained(W2V2_MODEL_NAME)
w2v2_model.to(DEVICE)  # type: ignore
w2v2_model.eval()


def process_embeddings(
    csv_path: Path,
    audio_dir: Path,
    id_column: str,
    output_cls: Path,
    output_mean: Path,
    errors_path: Path,
    split_name: str = "Split",
):
    df = pd.read_csv(csv_path)
    embeddings_cls = []
    embeddings_mean = []
    errors = {}

    processed_count = 0
    error_count = 0

    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_filename = row[id_column]
        audio_path = audio_dir / audio_filename

        try:
            audio = load_audio(audio_path)

            cls_emb, mean_emb = extract_embeddings(
                audio, w2v2_model, w2v2_processor
            )
            embeddings_cls.append(cls_emb)
            embeddings_mean.append(mean_emb)

        except Exception as e:
            errors[audio_filename] = str(e)
            error_count += 1
            print(f"[{split_name}] Error en {audio_filename}: {e}")

        processed_count += 1

        if processed_count % PRINT_EVERY == 0:
            print(
                f"[{split_name}] Procesados: {processed_count} audios | Fallos: {error_count}"
            )

    np.save(output_cls, np.array(embeddings_cls))
    np.save(output_mean, np.array(embeddings_mean))

    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[{split_name}] Tiempo total: {elapsed_time / 60:.2f} minutos")
