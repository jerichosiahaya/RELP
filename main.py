from store.store import Store
from config.env import VECTOR_PATH, EMBEDDING
from flask import Flask, request, jsonify
from wrapper.classification import Classification
from embedding.embedding import Embedding
from config.log import log

app = Flask(__name__)

logging = log("RELP")
log("werkzeug")

logging.info("Starting the app...")

embedding_model = Embedding(EMBEDDING)
embedding_store = Store(embedding=embedding_model.embedder, trained_vectors=VECTOR_PATH, top_k=3)
classification = Classification(embedding_store)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "content" not in data:
            return jsonify({"error": "Invalid input. 'content' field not found."}), 400

        article = data["content"]

        if not article:
            return jsonify({"error": "Invalid input. 'content' field is empty."}), 400

        prediction = classification.predict(article)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)