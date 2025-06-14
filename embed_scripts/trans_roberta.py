import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel  # type: ignore

# ================== CONFIGURACION ====================

MODEL_NAME = "PlanTL-GOB-ES/roberta-base-bne"
BATCH_SIZE = 32

# ======================================================

def encode_texts_pytorch(texts, model_name=MODEL_NAME, batch_size=BATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())

    final_embeddings = np.vstack(embeddings)
    return final_embeddings


def process_embeddings(csv_path: Path, output_path: Path, split_name: str = "Split"):
    df = pd.read_csv(csv_path)
    texts = df["transcription"].tolist()
    embeddings = encode_texts_pytorch(texts)
    np.save(output_path, embeddings)
    print(
        f"[{split_name}] Embeddings guardados en {output_path}."
    )

