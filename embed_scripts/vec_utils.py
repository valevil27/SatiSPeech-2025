from pathlib import Path
from fasttext.FastText import _FastText
import numpy as np
from numpy import ndarray
import fasttext
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModel # type: ignore
import torch
import gc

# Functions for loading embeddings

def load_fasttext_model(path_bin: Path) -> _FastText:
    """Loads a fasttext model from a binary file."""
    return fasttext.load_model(path_bin)


def load_npy_embeddings(vectors_path: Path, vocab_path: Path) -> tuple[ndarray, dict]:
    """Loads embeddings from a numpy file given the paths to the vectors and vocab files."""
    vectors: ndarray = np.load(vectors_path)
    vocab: ndarray = np.load(vocab_path)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return vectors, word_to_index

# Functions for transforming text to vectors

def text_to_vec_w2v(text: str, vectors: ndarray, word_to_index: dict) -> ndarray:
    """Transforms a text to a mean vector using word2vec."""
    words = text.split()
    vecs = [vectors[word_to_index[word]] for word in words if word in word_to_index]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(vectors.shape[1])


def text_to_vec_fasttext(text: str, fasttext_model: _FastText) -> ndarray:
    """Transforms a text to a mean vector using fasttext."""
    words = text.split()
    vectors = [fasttext_model.get_word_vector(word) for word in words]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(fasttext_model.get_dimension())


def text_to_vec_roberta(df_train: DataFrame, df_test: DataFrame, model_name="PlanTL-GOB-ES/roberta-base-bne") -> tuple[ndarray, ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    def embed_texts(texts):
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs) # noqa: F821
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return np.vstack(embeddings)
    X_train = embed_texts(df_train['texto_limpio'])
    X_test = embed_texts(df_test['texto_limpio'])
    del model
    gc.collect()
    return X_train, X_test
