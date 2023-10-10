import json
import gzip
import pickle
from tqdm import tqdm
from utils import formula as sim_formula
from config.log import log

class Store():
    def __init__(self, embedding, trained_vectors=None, top_k=5):
        self.embedder = embedding
        self.trained_vectors = trained_vectors
        self.top_k = top_k

    def load(self):
        with gzip.open(self.trained_vectors, "rb") as f:
            data = pickle.load(f)
        vectors = data["vectors"]
        documents = data["documents"]
        return documents, vectors

    def query(self, documents, vectors, query_text):
        query_vector = self.embedder(query_text)
        ranked_results, _ = sim_formula.hyper_svm_ranking_algorithm_sort(
            vectors, query_vector, top_k=self.top_k
        )
        return [documents[index] for index in ranked_results]
    
    # Only use this for training/vectorization phase
    def fit(self, data, key="text", save_name: str="data.pickle.gz"):
        docs = []
        vecs = []

        with open(data, "r") as f:
            entries = json.load(f)

        total_entries = len(entries)

        with tqdm(total=total_entries, desc="Processing and storing embeddings") as pbar:
            for entry in entries:
                docs.append(entry)
                v = self.embedder(entry[key])
                vecs.append(v)
                pbar.update(1)

        # save documents and its vectors
        self.save(vecs, docs, save_name)

    # Only use this for training/vectorization phase
    @staticmethod
    def save(vectors, documents, storage_file):
        data = {"vectors": vectors, "documents": documents}
        with gzip.open(storage_file, "wb") as f:
            pickle.dump(data, f)