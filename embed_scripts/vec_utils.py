# --- vectorization_utils.py ---

import numpy as np
import fasttext
from transformers import AutoTokenizer, AutoModel # type: ignore
import torch
import gc

# --- Funciones de carga de modelos ---

def load_fasttext_model(path_bin):
    """Carga un modelo FastText binario."""
    return fasttext.load_model(path_bin)


def load_npy_embeddings(vectors_path, vocab_path):
    """Carga embeddings y vocabulario desde archivos .npy."""
    vectors = np.load(vectors_path)
    vocab = np.load(vocab_path)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return vectors, word_to_index

# --- Funciones de transformación de texto a vectores ---

def text_to_vec_w2v(texto, vectors, word_to_index):
    """Transforma un texto a un vector promedio usando Word2Vec."""
    palabras = texto.split()
    vectores = [vectors[word_to_index[word]] for word in palabras if word in word_to_index]
    if vectores:
        return np.mean(vectores, axis=0)
    else:
        return np.zeros(vectors.shape[1])


def text_to_vec_fasttext(texto, modelo_fasttext):
    """Transforma un texto a un vector promedio usando FastText."""
    palabras = texto.split()
    vectores = [modelo_fasttext.get_word_vector(word) for word in palabras]
    if vectores:
        return np.mean(vectores, axis=0)
    else:
        return np.zeros(modelo_fasttext.get_dimension())


def text_to_vec_roberta(df_train, df_test, model_name="PlanTL-GOB-ES/roberta-base-bne"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    def embed_texts(texts):
        embeddings = []
        for texto in texts:
            inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs) # type: ignore
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
        return np.vstack(embeddings)
    X_train = embed_texts(df_train['texto_limpio'])
    X_test = embed_texts(df_test['texto_limpio'])
    del model
    gc.collect()
    return X_train, X_test

# --- Funciones para construir secuencias ---
def text_to_sequence_w2v(texto, word_to_index, embedding_matrix):
    palabras = texto.split()
    secuencia = []
    for word in palabras:
        if word in word_to_index:
            idx = word_to_index[word]
            secuencia.append(embedding_matrix[idx])
        else:
            secuencia.append(np.zeros(embedding_matrix.shape[1]))
    return np.array(secuencia)

def text_to_sequence_fasttext(texto, modelo_fasttext):
    palabras = texto.split()
    secuencia = []
    for word in palabras:
        vec = modelo_fasttext.get_word_vector(word)
        secuencia.append(vec)
    return np.array(secuencia)

def build_sequence_embedding(df_text, build_fn, maxlen=50):
    secuencias = [build_fn(texto) for texto in df_text]
    secuencias_padded = []
    for seq in secuencias:
        if len(seq) >= maxlen:
            seq = seq[:maxlen]
        else:
            padding = np.zeros((maxlen - len(seq), seq.shape[1]))
            seq = np.vstack([seq, padding])
        secuencias_padded.append(seq)
    return np.stack(secuencias_padded)
