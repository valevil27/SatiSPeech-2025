# fusion_utils.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Loader de embeddings

def load_embeddings_npy(file_path, idx_train, idx_val, normalize=True):
    embeddings = np.load(file_path)
    X_train = embeddings[idx_train]
    X_val = embeddings[idx_val]
    del embeddings
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
    return X_train, X_val, scaler

# Funciones de fusión de embeddings

def fusion_concat(embedding_a, embedding_b):
    """
    Fusión por concatenación de dos embeddings.
    """
    return np.concatenate([embedding_a, embedding_b], axis=1)

def normalize(embedding_a, embedding_b):
    print("Normalizing embeddings...")
    dim_a, dim_b = embedding_a.shape[1], embedding_b.shape[1]
    if dim_a == dim_b:
        return embedding_a, embedding_b
    min_dim = min(dim_a, dim_b)
    emb_a = embedding_a[:, :min_dim]
    emb_b = embedding_b[:, :min_dim]
    print(f"Embedding a: {emb_a.shape}")
    print(f"Embedding b: {emb_b.shape}")
    return emb_a, emb_b

def fusion_mean(embedding_a, embedding_b):
    """
    Fusión por media entre dos embeddings del mismo tamaño.
    """
    emb_a, emb_b = normalize(embedding_a, embedding_b)
    return (emb_a + emb_b) / 2

def fusion_weighted(embedding_a, embedding_b, weight_a=0.5, weight_b=0.5):
    """
    Fusión ponderada entre dos embeddings del mismo tamaño.
    """
    embedding_a, embedding_b = normalize(embedding_a, embedding_b)
    return (weight_a * embedding_a) + (weight_b * embedding_b)

def fusion_attention(embedding_a, embedding_b, num_heads=4):
    """
    Fusión mediante atención multi-cabeza entre dos embeddings.
    """
    embedding_a, embedding_b = normalize(embedding_a, embedding_b)
    d_model = embedding_a.shape[1]
    key_dim = d_model // num_heads

    mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    tensor_a = tf.convert_to_tensor(embedding_a, dtype=tf.float32)
    tensor_b = tf.convert_to_tensor(embedding_b, dtype=tf.float32)

    tensor_a = tf.expand_dims(tensor_a, axis=1)
    tensor_b = tf.expand_dims(tensor_b, axis=1)

    att_output = mha_layer(query=tensor_a, value=tensor_b, key=tensor_b)
    att_output = tf.squeeze(att_output, axis=1).numpy()

    scaler = StandardScaler()
    att_output = scaler.fit_transform(att_output)

    return att_output

def search_best_weighted_fusion(X_train_a, X_train_b, X_val_a, X_val_b, y_train, y_val, model=None):
    """
    Búsqueda de la mejor combinación de pesos para la fusión ponderada.
    """
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=42)

    best_score = 0
    best_weights = (0.5, 0.5)

    for w_a in np.arange(0.1, 1.0, 0.1):
        w_b = 1.0 - w_a
        X_train_fusion = fusion_weighted(X_train_a, X_train_b, weight_a=w_a, weight_b=w_b)
        X_val_fusion = fusion_weighted(X_val_a, X_val_b, weight_a=w_a, weight_b=w_b)

        model.fit(X_train_fusion, y_train)
        score = model.score(X_val_fusion, y_val)

        if score > best_score:
            best_score = score
            best_weights = (w_a, w_b)

    return best_weights, best_score