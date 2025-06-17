# verify_roberta_embeddings.py

import os
import numpy as np

# ================== CONFIGURACION ====================

DATA_DIR = "./data/public_data/roberta_embeddings"
TRAIN_EMBEDS_PATH = os.path.join(DATA_DIR, "train_roberta_embeddings.npy")
TEST_EMBEDS_PATH = os.path.join(DATA_DIR, "test_roberta_embeddings.npy")

# ======================================================

# Funciones auxiliares

def verify_embeddings(path, expected_samples=None, embedding_dim=None):
    if not os.path.exists(path):
        print(f"❌ No encontrado: {path}")
        return

    embeds = np.load(path)
    print(f"✅ {os.path.basename(path)} cargado. Forma: {embeds.shape}")

    if expected_samples is not None and embeds.shape[0] != expected_samples:
        print(f"⚠️ Advertencia: Se esperaban {expected_samples} muestras, pero hay {embeds.shape[0]}")
    else:
        print("✔️ Número de muestras correcto.")

    if embedding_dim is not None and embeds.shape[1] != embedding_dim:
        print(f"⚠️ Advertencia: Se esperaba tamaño de embedding {embedding_dim}, pero es {embeds.shape[1]}")
    else:
        print("✔️ Dimensión de embedding correcta.")

# =============== VERIFICACION PRINCIPAL ================

if __name__ == "__main__":
    print("Verificando embeddings...")

    verify_embeddings(TRAIN_EMBEDS_PATH, expected_samples=6000, embedding_dim=768)  # Ajusta el número si sabes el real
    verify_embeddings(TEST_EMBEDS_PATH, expected_samples=2000, embedding_dim=768)   # Ajusta el número si sabes el real

    print("\n✅ Verificación completa.")
