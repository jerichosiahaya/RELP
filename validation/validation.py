import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from embedding.embedding import Embedding
from store.store import Store
from wrapper.classification import Classification

import json
from tqdm import tqdm

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

print("Starting the validation...")

# Path to the vectorized data
VECTOR_PATH=""

# Choose the embedding, should be the same with vectorized data
EMBEDDING="indobert"

# Load the testing dataset
TESTING_DATASET_PATH=""

with open(TESTING_DATASET_PATH, "r") as f:
    data = json.load(f)


embedding_model = Embedding(EMBEDDING)
embedding_store = Store(embedding=embedding_model.embedder, trained_vectors=VECTOR_PATH, top_k=3)
classification = Classification(embedding_store)

true_labels = []
predicted_labels = []

total_entries = len(data)

for entry in tqdm(data, total=total_entries, desc="Classifying"):
    true_label = entry["label"]
    predicted_label = classification.classify(entry["text"])

    true_labels.append(true_label)
    predicted_labels.append(predicted_label)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Compute F1 score
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Compute accuracy score
accuracy = accuracy_score(true_labels, predicted_labels)

print("F1 Score:", f1)
print("Accuracy Score:", accuracy)