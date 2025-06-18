from typing import Callable
import numpy as np
import keras_tuner as kt
from keras_tuner import HyperParameters
from keras import Sequential

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
                    units=hp.Int(
                        f"units_{i}", min_value=32, max_value=256, step=32
                    ),
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
