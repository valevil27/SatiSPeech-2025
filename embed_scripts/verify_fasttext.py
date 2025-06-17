# verify_fasttext_embeddings.py

import os
import numpy as np

DATA_DIR = "./data/public_data/fasttext_embeddings"
TRAIN_EMBEDS = os.path.join(DATA_DIR, "train_fasttext_embeddings.npy")
TEST_EMBEDS = os.path.join(DATA_DIR, "test_fasttext_embeddings.npy")

def verificar_embeddings(path, expected_samples=None, embedding_dim=300):
    if not os.path.exists(path):
        print(f"❌ Archivo no encontrado: {path}")
        return

    embeds = np.load(path)
    print(f"✅ {os.path.basename(path)} cargado. Forma: {embeds.shape}")

    if expected_samples is not None and embeds.shape[0] != expected_samples:
        print(f"⚠️ Esperado {expected_samples} muestras, obtenido: {embeds.shape[0]}")
    if embeds.shape[1] != embedding_dim:
        print(f"⚠️ Dimensión incorrecta: se esperaba {embedding_dim}, se obtuvo {embeds.shape[1]}")
    else:
        print("✔️ Dimensiones correctas.")

if __name__ == "__main__":
    verificar_embeddings(TRAIN_EMBEDS, expected_samples=6000)
    verificar_embeddings(TEST_EMBEDS, expected_samples=2000)
    print("\n✅ Verificación de embeddings completada.")