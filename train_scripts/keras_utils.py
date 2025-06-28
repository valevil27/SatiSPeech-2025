from typing import Callable
import numpy as np
import keras_tuner as kt
import keras
from keras_tuner import HyperParameters
from keras import Sequential, Model

ModelBuilder = Callable[[HyperParameters], Sequential]


def build_model(X_train: np.ndarray, y_train: np.ndarray) -> ModelBuilder:
    """
    Creates a model builder function for Keras Tuner hyperparameter search.

    This function returns a callable that builds a Keras Sequential model with a variable number of dense layers,
    units, activations, and dropout rates, as specified by the provided HyperParameters object. The output layer
    uses softmax activation and the number of units equal to the number of unique classes in y_train.

    Args:
            X_train (np.ndarray): Training feature data, used to determine input shape.
            y_train (np.ndarray): Training labels, used to determine the number of output classes.

    Returns:
            ModelBuilder: A function that takes a HyperParameters object and returns a compiled Keras Sequential model.
    """
    from keras import layers, optimizers

    def get_model(hp: HyperParameters) -> Sequential:
        model: Sequential = Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))

        # Número de capas ocultas y unidades
        for i in range(hp.Int("num_layers", 1, 3)):  # type: ignore
            model.add(
                layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                    activation=hp.Choice("activation", ["relu", "tanh"]),
                )
            )
            model.add(
                layers.Dropout(
                    rate=hp.Float(
                        f"dropout_{i}", min_value=0.1, max_value=0.5, step=0.1
                    )
                )
            )

        model.add(layers.Dense(len(set(y_train)), activation="softmax"))

        model.compile(
            optimizer=optimizers.Adam(
                hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")  # type: ignore
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    return get_model


def get_tuner(
    model_builder: ModelBuilder, project_name: str, seed: int
) -> kt.RandomSearch:
    """
    Creates and returns a Keras Tuner RandomSearch object for hyperparameter tuning.

    Args:
            model_builder (ModelBuilder): A function or callable that builds and returns a compiled Keras model.
            project_name (str): The name of the project. Used to identify tuning results.
            seed (int): Random seed for reproducibility.

    Returns:
            kt.RandomSearch: An instance of Keras Tuner's RandomSearch configured with the given parameters.
    """
    return kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=30,
        seed=seed,
        executions_per_trial=1,
        directory="keras_tuner_dir",
        project_name=project_name,
    )


def get_early_stop():
    from keras import callbacks

    return callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )


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

    from keras import layers, models, optimizers

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


def get_attention_embeddings(
    model: keras.Sequential,
    text: np.ndarray,
    audio: np.ndarray,
) -> np.ndarray:
    emb_model = Model(
        inputs=model.inputs, outputs=model.get_layer("fusion_output").output
    )
    return emb_model.predict([text, audio], verbose=0)  # type: ignore
