import ctranslate2
import transformers
import numpy as np
import torch
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from config.env import EMBEDDING_OPENAI_API_TYPE, EMBEDDING_OPENAI_API_KEY, EMBEDDING_OPENAI_API_BASE, EMBEDDING_OPENAI_API_VERSION, EMBEDDING_OPENAI_API_DEPLOYMENT_NAME, MODEL_PATH, TOKENIZER_PATH

device = "cpu"

class Embedding():
    def __init__(self, embedding_model_choice):
        if embedding_model_choice == "indobert":
            self.embedder = self.indobert_embedder
        else:
            self.embedder = self.openai_embedder

    def indobert_embedder(self, text: str):
        encoder = ctranslate2.Encoder(MODEL_PATH, device)

        tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        inputs = [text]

        tokens = tokenizer(inputs, truncation='longest_first', max_length=512, padding='longest').input_ids

        output = encoder.forward_batch(tokens)
        pooler_output = output.pooler_output

        pooler_output = np.array(pooler_output)
        pooler_output = torch.as_tensor(pooler_output)

        formatted_vector = [float(val.item()) for val in pooler_output[0]]

        return formatted_vector

    def openai_embedder(self, text: str):
        embeddings = OpenAIEmbeddings(
            deployment=EMBEDDING_OPENAI_API_DEPLOYMENT_NAME,
            openai_api_version=EMBEDDING_OPENAI_API_VERSION,
            openai_api_key=EMBEDDING_OPENAI_API_KEY,
            openai_api_base=EMBEDDING_OPENAI_API_BASE,
            openai_api_type=EMBEDDING_OPENAI_API_TYPE
        )

        result = embeddings.embed_query(text)

        return result