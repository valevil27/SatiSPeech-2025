# trans_wav2vec2BERT.py
import traceback
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
import time
from tqdm import tqdm
from transformers.models.wav2vec2_bert import Wav2Vec2BertModel
from transformers.models.seamless_m4t import SeamlessM4TFeatureExtractor

# =============== CONFIGURACIÓN =========================
W2V2BERT_MODEL_NAME = "facebook/w2v-bert-2.0"  # 600 M params, 4.5 M h audio  :contentReference[oaicite:0]{index=0}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SAMPLING_RATE = 16_000
PRINT_EVERY = 100

extractor = SeamlessM4TFeatureExtractor.from_pretrained(W2V2BERT_MODEL_NAME)
model = Wav2Vec2BertModel.from_pretrained(W2V2BERT_MODEL_NAME)
model.to(DEVICE)  # type: ignore
model.eval()


def process_embeddings(
    csv_path: Path,
    audio_dir: Path,
    id_column: str,
    output_cls: Path,
    output_mean: Path,
    errors_path: Path,
    split_name: str = "Split",
):
    """
    Lee el CSV, procesa cada archivo de audio y guarda:
      • output_cls.npy  → vector CLS por audio
      • output_mean.npy → media temporal de los embeddings
    """
    df = pd.read_csv(csv_path)
    emb_cls, emb_mean, errors = [], [], {}
    processed, failed = 0, 0
    start = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        audio_path = audio_dir / row[id_column]
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != TARGET_SAMPLING_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, sr, TARGET_SAMPLING_RATE
                )
            audio = waveform.squeeze().numpy()
            inputs = extractor(
                [audio],
                sampling_rate=TARGET_SAMPLING_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            cls = hidden[:, 0].squeeze(0).cpu().numpy()
            mean = hidden.mean(dim=1).squeeze(0).cpu().numpy()
            emb_cls.append(cls)
            emb_mean.append(mean)
        except Exception as e:
            errors[row[id_column]] = str(e)
            failed += 1
            print(f"[{split_name}] Error en {row[id_column]}: {repr(e)}")
            traceback.print_exc()
        processed += 1
        if processed % PRINT_EVERY == 0:
            print(f"[{split_name}] Procesados: {processed} | Fallos: {failed}")

    np.save(output_cls, np.asarray(emb_cls))
    np.save(output_mean, np.asarray(emb_mean))
    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=4)

    mins = (time.time() - start) / 60
    print(f"[{split_name}] Tiempo total: {mins:.2f} min")
