# trans_fasttext_embeddings.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vec_utils import cargar_fasttext_model, texto_a_vector_fasttext

# ================== CONFIGURACIÓN ====================

nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

DATA_DIR = Path("./data/public_data")
CSV_TRAIN = DATA_DIR / "SatiSPeech_phase_2_train_public.csv"
CSV_TEST = DATA_DIR / "SatiSPeech_phase_2_test_public.csv"

OUTPUT_DIR = DATA_DIR / "fasttext_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDS_TRAIN = OUTPUT_DIR / "train_fasttext_embeddings.npy"
EMBEDS_TEST = OUTPUT_DIR / "test_fasttext_embeddings.npy"

FASTTEXT_MODEL_PATH = "./cc.es.300.bin"

# =====================================================


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\\S+|www\\S+|https\\S+", "", texto)
    texto = re.sub(r"\\@\\w+|\\#", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    texto = re.sub(r"\\s+", " ", texto).strip()
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return " ".join(palabras)


def procesar_embeddings(
    csv_path: Path, output_path: Path, split_name: str = "Split"
):
    print(f"[{split_name}] Procesando {csv_path}...")
    df = pd.read_csv(csv_path)
    df["texto_limpio"] = df["transcription"].apply(limpiar_texto)

    modelo_fasttext = cargar_fasttext_model(FASTTEXT_MODEL_PATH)

    X = np.vstack(
        df["texto_limpio"].apply(
            lambda x: texto_a_vector_fasttext(x, modelo_fasttext) # type: ignore
        ) # type: ignore
    )
    np.save(output_path, X)

    print(
        f"[{split_name}] Embeddings guardados en {output_path} con forma {X.shape}."
    )


# =============== PROCESAMIENTO PRINCIPAL ================

if __name__ == "__main__":
    procesar_embeddings(CSV_TRAIN, EMBEDS_TRAIN, split_name="Train")
    procesar_embeddings(CSV_TEST, EMBEDS_TEST, split_name="Test")
    print("\n✅ Embeddings FastText generados correctamente.")
