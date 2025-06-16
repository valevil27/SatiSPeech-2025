# --- vectorization_utils.py ---

import numpy as np
import fasttext
from transformers import AutoTokenizer, AutoModel # type: ignore
import torch
import gc

# Functions for loading embeddings

def load_fasttext_model(path_bin):
    """Loads a fasttext model from a binary file."""
    return fasttext.load_model(path_bin)


def load_npy_embeddings(vectors_path, vocab_path):
    """Loads embeddings from a numpy file given the paths to the vectors and vocab files."""
    vectors = np.load(vectors_path)
    vocab = np.load(vocab_path)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    return vectors, word_to_index

# Functions for transforming text to vectors

def text_to_vec_w2v(text, vectors, word_to_index):
    """Transforms a text to a mean vector using word2vec."""
    words = text.split()
    vecs = [vectors[word_to_index[word]] for word in words if word in word_to_index]
    if vecs:
        return np.mean(vecs, axis=0)
    else:
        return np.zeros(vectors.shape[1])


def text_to_vec_fasttext(text, fasttext_model):
    """Transforms a text to a mean vector using fasttext."""
    words = text.split()
    vectors = [fasttext_model.get_word_vector(word) for word in words]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(fasttext_model.get_dimension())


def text_to_vec_roberta(df_train, df_test, model_name="PlanTL-GOB-ES/roberta-base-bne"):
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

# Functions for transforming text to sequences
def text_to_sequence_w2v(text, word_to_index, embedding_matrix):
    """Transforms a text to a sequence of word vectors using word2vec, for training RRNs like GRU and LSTM."""
    words = text.split()
    sequence = []
    for word in words:
        if word in word_to_index:
            idx = word_to_index[word]
            sequence.append(embedding_matrix[idx])
        else:
            sequence.append(np.zeros(embedding_matrix.shape[1]))
    return np.array(sequence)

def text_to_sequence_fasttext(text, model_fasttext):
    """Transforms a text to a sequence of word vectors using fasttext, for training RRNs like GRU and LSTM."""
    words = text.split()
    sequence = []
    for word in words:
        vec = model_fasttext.get_word_vector(word)
        sequence.append(vec)
    return np.array(sequence)

def build_sequence_embedding(df_text, build_fn, maxlen=50):
    """ Builds a sequence of embeddings given a dataframe with texts and a function to build the embeddings. """
    sequences = [build_fn(text) for text in df_text]
    sequences_padded = []
    for seq in sequences:
        if len(seq) >= maxlen:
            seq = seq[:maxlen]
        else:
            padding = np.zeros((maxlen - len(seq), seq.shape[1]))
            seq = np.vstack([seq, padding])
        sequences_padded.append(seq)
    return np.stack(sequences_padded)
