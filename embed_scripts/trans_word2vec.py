from pathlib import Path
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vec_utils import load_npy_embeddings, text_to_vec_w2v

# ================== CONFIGURACIÓN ====================

nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

W2V_VECTORS = "./word2vec_vectors.npy"
W2V_VOCAB = "./word2vec_vocab.npy"

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


def process_embeddings(csv_path: Path, output_path: Path, split_name="Split"):
    print(f"[{split_name}] Procesando {csv_path}...")
    df = pd.read_csv(csv_path)
    df["texto_limpio"] = df["transcription"].apply(limpiar_texto)

    vectors, vocab = load_npy_embeddings(W2V_VECTORS, W2V_VOCAB)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    X = np.vstack(
        df["texto_limpio"].apply(
            lambda x: text_to_vec_w2v(x, vectors, word_to_index) # type: ignore
        ) # type: ignore
    )
    np.save(output_path, X)

    print(
        f"[{split_name}] Embeddings guardados en {output_path} con forma {X.shape}."
    )
