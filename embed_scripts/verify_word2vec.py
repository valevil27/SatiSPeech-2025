# verify_word2vec_embeddings.py

import numpy as np
import pandas as pd
import os

# --- Paths ---
DATA_DIR = "./data/public_data"
CSV_TRAIN = os.path.join(DATA_DIR, "SatiSPeech_phase_2_train_public.csv")
CSV_TEST = os.path.join(DATA_DIR, "SatiSPeech_phase_2_test_public.csv")

EMBEDS_TRAIN = os.path.join(DATA_DIR, "word2vec_embeddings/train_word2vec_embeddings.npy")
EMBEDS_TEST = os.path.join(DATA_DIR, "word2vec_embeddings/test_word2vec_embeddings.npy")

# --- Cargar datos y embeddings ---
df_train = pd.read_csv(CSV_TRAIN)
df_test = pd.read_csv(CSV_TEST)

X_train = np.load(EMBEDS_TRAIN)
X_test = np.load(EMBEDS_TEST)

# --- Verificaciones ---
print("🔍 Verificando consistencia de los Word2Vec embeddings:\n")

print(f"- train.csv contiene {len(df_train)} filas")
print(f"- test.csv contiene {len(df_test)} filas")
print(f"- Embeddings de train: {X_train.shape}")
print(f"- Embeddings de test: {X_test.shape}")

assert len(df_train) == X_train.shape[0], "❌ Mismatch en número de ejemplos (train)"
assert len(df_test) == X_test.shape[0], "❌ Mismatch en número de ejemplos (test)"
assert X_train.shape[1] == 300, "❌ Dimensión incorrecta de embedding (esperado: 300)"
assert X_test.shape[1] == 300, "❌ Dimensión incorrecta de embedding (esperado: 300)"

print("\n✅ Verificación completada correctamente. Los embeddings son válidos.")

