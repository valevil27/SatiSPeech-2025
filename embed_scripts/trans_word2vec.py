from pathlib import Path
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from vec_utils import load_npy_embeddings, text_to_vec_w2v
from trans_fasttext import clean_text

# ================== CONFIGURACIÓN ====================

nltk.download("stopwords")
stop_words = set(stopwords.words("spanish"))

W2V_VECTORS = Path("./embedding_files/word2vec_vectors.npy")
W2V_VOCAB = Path("./embedding_files/word2vec_vocab.npy")

# =====================================================

def process_embeddings(csv_path: Path, output_path: Path, split_name="Split"):
    print(f"[{split_name}] Processing {csv_path}...")
    df = pd.read_csv(csv_path)
    df["clean_text"] = df["transcription"].apply(clean_text)

    vectors, vocab = load_npy_embeddings(W2V_VECTORS, W2V_VOCAB)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    X = np.vstack(
        df["clean_text"].apply(
            lambda x: text_to_vec_w2v(x, vectors, word_to_index) # type: ignore
        ) # type: ignore
    )
    np.save(output_path, X)

    print(
        f"[{split_name}] Embeddings saved to {output_path} with shape {X.shape}."
    )
