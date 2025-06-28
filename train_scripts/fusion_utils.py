from pathlib import Path
from typing import Any, Optional
import keras
import numpy as np
from keras import layers, models, optimizers, Model
from keras_tuner import HyperParameters
from keras_utils import ModelBuilder, get_early_stop, get_tuner
from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


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


def normalize(
    embedding_a: ndarray, embedding_b: ndarray, verbose: int = 0
) -> tuple[ndarray, ndarray]:
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
    if verbose > 0:
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


def train_attention(
    X_train_text: np.ndarray,
    X_train_audio: np.ndarray,
    X_val_text: np.ndarray,
    X_val_audio: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    name: str,
    random_state: int,
) -> tuple[dict[str, Any], keras.Model]:
    """Tunes and trains an attention-based model"""
    print("\nTuning attention-based model...")
    builder = build_attention_model(
        X_train_text.shape[1],
        X_train_audio.shape[1],
        y_train.shape[0],
    )
    tuner = get_tuner(builder, name, random_state)
    tuner.search(
        [X_train_text, X_train_audio],
        y_train,
        epochs=30,
        validation_split=0.2,
        callbacks=[get_early_stop()],
        verbose=0,  # type: ignore
    )
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = builder(best_hps)
    best_model.fit(
        [X_train_text, X_train_audio],
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[get_early_stop()],
        verbose=0,  # type: ignore
    )
    y_pred = best_model.predict([X_val_text, X_val_audio])
    y_pred_classes = y_pred.argmax(axis=1)
    report = classification_report(y_val, y_pred_classes, output_dict=True)
    assert isinstance(report, dict)
    results = {"ATTENTION": report}
    results["ATTENTION"]["Hyperparameters"] = best_hps.values
    print("#### Report for Attention Fusion:\n", report)
    return results, best_model


def build_attention_model(
    text_dim: int, audio_dim: int, y_train: np.ndarray
) -> ModelBuilder:
    """Returns a model builder that applies cross attention between text and
    audio embeddings.

    The returned callable builds a Keras model that projects both embeddings to
    the same dimension, applies ``MultiHeadAttention`` and concatenates the
    attention output with the text projection before the classification layers.
    ``keras_tuner`` will explore the dimension size, number of heads and other
    dense layer parameters.
    """

    def get_model(hp: HyperParameters):
        nclasses = 2
        d_model = hp.Choice("d_model", [64, 128, 256])
        num_heads = 4
        assert isinstance(d_model, int)
        key_dim = d_model // num_heads

        text_in = layers.Input(shape=(text_dim,), name="text_input")
        audio_in = layers.Input(shape=(audio_dim,), name="audio_input")
        text_proj = layers.Dense(d_model)(text_in)
        audio_proj = layers.Dense(d_model)(audio_in)

        query = layers.Reshape((1, d_model))(text_proj)
        value = layers.Reshape((1, d_model))(audio_proj)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
            query=query, value=value, key=value
        )
        attn = layers.Reshape((d_model,))(attn)
        fused = layers.Concatenate()([text_proj, attn])

        x = layers.Dense(
            hp.Choice("units", [64, 128, 256]),
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )(fused)
        x = layers.Dropout(
            hp.Float("dropout", 0.1, 0.5, step=0.1), name="fusion_output"
        )(x)
        out = layers.Dense(nclasses, activation="softmax")(x)

        model = models.Model(inputs=[text_in, audio_in], outputs=out)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        assert isinstance(learning_rate, float)
        model.compile(
            optimizer=optimizers.Adam(learning_rate),  # type: ignore
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    return get_model


def fusion_attention(
    text: np.ndarray,
    audio: np.ndarray,
    model: keras.Sequential,
) -> np.ndarray:
    emb_model = Model(
        inputs=model.inputs, outputs=model.get_layer("fusion_output").output
    )
    return emb_model.predict([text, audio], verbose=0)  # type: ignore
