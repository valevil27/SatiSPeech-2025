import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import HubertModel  # type: ignore

# ================== CONFIGURACION ====================

# Modelo
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"

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

hubert_model = HubertModel.from_pretrained(HUBERT_MODEL_NAME)
hubert_model.to(DEVICE)  # type: ignore
hubert_model.eval()


# Funciones auxiliares
def load_audio(path, target_sr=16000):
    waveform, sr = torchaudio.load(path)
    # Convertir a mono si es estéreo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # convertir a 1D tensor


def extract_embeddings(audio_tensor, model, processor=None):
    with torch.no_grad():
        if processor is not None:
            inputs = processor(
                audio_tensor,
                sampling_rate=TARGET_SAMPLING_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(DEVICE)
        else:
            input_values = audio_tensor.unsqueeze(0).to(DEVICE)
        outputs = model(input_values)
        # (time_steps, hidden_dim)
        hidden_states = outputs.last_hidden_state.squeeze(0)
        cls_embedding = hidden_states[0].cpu().numpy()
        mean_embedding = hidden_states.mean(dim=0).cpu().numpy()
        return cls_embedding, mean_embedding


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

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        audio_filename = row[id_column]
        audio_path = audio_dir / audio_filename

        try:
            audio = load_audio(audio_path)
            cls_emb_hubert, mean_emb_hubert = extract_embeddings(audio, hubert_model)
            embeddings_cls.append(cls_emb_hubert)
            embeddings_mean.append(mean_emb_hubert)

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
