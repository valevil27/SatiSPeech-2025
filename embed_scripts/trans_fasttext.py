from pathlib import Path
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vec_utils import load_fasttext_model, text_to_vec_fasttext

# ================== CONFIGURACIÓN ====================

nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))
FASTTEXT_MODEL_PATH = Path("./embedding_files/cc.es.300.bin")

# =====================================================


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", "", text)
    text = re.sub(r"\\@\\w+|\\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    words = text.split()
    words = [p for p in words if p not in stop_words]
    return " ".join(words)


def process_embeddings(
    csv_path: Path, output_path: Path, split_name: str = "Split"
):
    print(f"[{split_name}] Procesando {csv_path}...")
    df = pd.read_csv(csv_path)
    df["texto_limpio"] = df["transcription"].apply(clean_text)

    modelo_fasttext = load_fasttext_model(FASTTEXT_MODEL_PATH)

    X = np.vstack(
        df["texto_limpio"].apply(
            lambda x: text_to_vec_fasttext(x, modelo_fasttext)  # type: ignore
        )  # type: ignore
    )
    np.save(output_path, X)

    print(
        f"[{split_name}] Embeddings guardados en {output_path} con forma {X.shape}."
    )
