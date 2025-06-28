from pathlib import Path
import numpy as np
from numpy import ndarray
import tensorflow as tf
from typing import Optional
from tensorflow.keras.layers import MultiHeadAttention  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Embedding loader


def load_embeddings_npy(
    file_path: Path,
    idx_train: ndarray,
    idx_val: ndarray,
    normalize: bool = True,
) -> tuple[ndarray, ndarray, Optional[StandardScaler]]:
    """
    Loads the embeddings from a .npy file and returns the train and validation sets.
    If normalize is True, the embeddings are normalized using StandardScaler and the scaler is returned, otherwise None.
    """
    scaler = None
    embeddings = np.load(file_path)
    X_train = embeddings[idx_train]
    X_val = embeddings[idx_val]
    del embeddings
    if normalize:
        scaler = StandardScaler()
        X_train: ndarray = scaler.fit_transform(X_train)
        X_val: ndarray = scaler.transform(X_val)  # type: ignore
    return X_train, X_val, scaler


# Embedding fusion functions


def fusion_concat(embedding_a: ndarray, embedding_b: ndarray) -> ndarray:
    """
    Concatenation fusion between two embeddings.
    """
    return np.concatenate([embedding_a, embedding_b], axis=1)


def normalize(embedding_a: ndarray, embedding_b: ndarray) -> tuple[ndarray, ndarray]:
    """
    Given two embeddings, normalizes them to the same size, making them able to be fused with
    several fusion methods that require them to be of the same size.
    """
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


def fusion_mean(embedding_a: ndarray, embedding_b: ndarray) -> ndarray:
    """
    Mean fusion between two embeddings. Requires the embeddings to be of the same size.
    """
    emb_a, emb_b = normalize(embedding_a, embedding_b)
    return (emb_a + emb_b) / 2


def fusion_weighted(
    embedding_a: ndarray,
    embedding_b: ndarray,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> ndarray:
    """
    Weighted fusion between two embeddings. Requires the embeddings to be of the same size.
    """
    embedding_a, embedding_b = normalize(embedding_a, embedding_b)
    return (weight_a * embedding_a) + (weight_b * embedding_b)


def fusion_attention(
    embedding_a: ndarray, embedding_b: ndarray, num_heads: int = 4
) -> ndarray:
    """
    Multihead attention fusion between two embeddings. Requires the embeddings to be of the same size.
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


def search_best_weighted_fusion(
    X_train_a: ndarray,
    X_train_b: ndarray,
    X_val_a: ndarray,
    X_val_b: ndarray,
    y_train: ndarray,
    y_val: ndarray,
    model: Optional[LogisticRegression] = None,
) -> tuple[tuple[float, float], float]:
    """
    Search for the best fusion weight between two embeddings using a logistic regression model.
    """
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=420)

    best_score: float = 0
    best_weights = (0.5, 0.5)

    for w_a in np.arange(0.1, 1.0, 0.1):
        w_b = 1.0 - w_a
        X_train_fusion = fusion_weighted(
            X_train_a, X_train_b, weight_a=w_a, weight_b=w_b
        )
        X_val_fusion = fusion_weighted(X_val_a, X_val_b, weight_a=w_a, weight_b=w_b)

        model.fit(X_train_fusion, y_train)
        score: float = float(model.score(X_val_fusion, y_val))

        if score > best_score:
            best_score = score
            best_weights = (w_a, w_b)

    return best_weights, best_score
